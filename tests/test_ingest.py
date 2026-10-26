import gzip
import tempfile
import unittest
from pathlib import Path

from vibiomir.ingest.domains import domain_language, rate_group
from vibiomir.ingest.extract import decode_html, element_text, extract_html, extract_page
from vibiomir.ingest.state import CrawlState, PageUpdate

try:
    import httpx
    import lxml.html
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - ingest extra not installed
    httpx = None


class DomainTest(unittest.TestCase):
    def test_language_comes_from_domain_table(self):
        self.assertEqual(domain_language("pubmed.ncbi.nlm.nih.gov"), "en")
        self.assertEqual(domain_language("ask.39.net"), "zh")
        self.assertEqual(domain_language("test.pmphai.com"), "zh")
        self.assertEqual(domain_language("www.baidu.com"), "zh")
        self.assertEqual(domain_language("suckhoedoisong.vn"), "vi")
        self.assertEqual(domain_language("phunusuckhoe.giadinhonline.vn"), "vi")

    def test_subdomains_share_a_rate_group(self):
        self.assertEqual(rate_group("care.39.net"), "39.net")
        self.assertEqual(rate_group("ask.39.net"), "39.net")
        self.assertEqual(rate_group("baidianfeng.familydoctor.com.cn"), "familydoctor.com.cn")
        self.assertEqual(rate_group("phuyen.baodaklak.vn"), "baodaklak.vn")
        self.assertEqual(rate_group("bachmai.gov.vn"), "bachmai.gov.vn")


@unittest.skipIf(httpx is None, "ingest extra not installed")
class ExtractTest(unittest.TestCase):
    def test_block_elements_never_glue_tokens(self):
        element = lxml.html.fragment_fromstring(
            "<div><p>A</p><p>B<br>C</p><table><tr><td>x</td><td>y</td></tr></table>"
            "<span>in</span><b>line</b></div>"
        )
        self.assertEqual(element_text(element), "A\n\nB\nC\n\nx y\n\ninline")

    def test_meta_gb2312_is_decoded_as_gb18030(self):
        body = '<html><head><meta charset="gb2312"></head><body><h1>乙脑疫苗</h1></body>'
        html, encoding = decode_html(body.encode("gb2312"), "text/html")
        self.assertEqual(encoding, "gb18030")
        self.assertIn("乙脑疫苗", html)

    def test_header_charset_list_takes_first_entry(self):
        body = "<p>Sức khỏe</p>".encode("utf-8")
        html, encoding = decode_html(body, "text/html; charset=utf-8,gbk")
        self.assertEqual(encoding, "utf-8")
        self.assertIn("Sức khỏe", html)

    def test_site_rule_keeps_question_and_answer(self):
        html = """<html><body><div class="cont_l"><div id="sub">首页 &gt; 内科</div>
        <div class="ask_cont"><h1 class="ask_tit">久咳是怎么回事</h1>
        <div class="ask_hid"><p class="txt_ms">咳嗽两个月</p>
        <p class="txt_bc">补充说明：无</p></div>
        <p class="txt_label"><span>标签</span></p></div>
        <div class="selected"><div class="doctor_all">医生 主治医师</div>
        <p class="sele_txt">久咳可能是慢性支气管炎。</p></div>
        <div class="relevant_ask">相关问题</div></div></body></html>"""
        extracted = extract_html(html, "ask.39.net")
        self.assertEqual(extracted.extractor, "site:ask.39.net")
        expected = "久咳是怎么回事\n\n咳嗽两个月\n\n补充说明：无\n\n久咳可能是慢性支气管炎。"
        self.assertEqual(extracted.text, expected)

    def test_generic_cms_rule_and_title_matching_og(self):
        body = "Nội dung bài viết về tăng huyết áp. " * 10
        html = f"""<html><head><meta property="og:title" content="Tăng huyết áp | Báo"></head>
        <body><h1>Logo báo</h1><h1>Tăng huyết áp</h1>
        <div class="detail-sapo">Sapo ngắn.</div>
        <div class="detail-content"><p>{body}</p><script>var x = 1;</script></div>
        <div class="related">Bài liên quan</div></body></html>"""
        extracted = extract_html(html, "unknown-news.vn")
        self.assertEqual(extracted.extractor, "cms:detail-content")
        self.assertEqual(extracted.title, "Tăng huyết áp")
        self.assertTrue(extracted.text.startswith("Tăng huyết áp\n\nSapo ngắn.\n\nNội dung"))
        self.assertNotIn("Bài liên quan", extracted.text)
        self.assertNotIn("var x", extracted.text)

    def test_stray_closing_html_tag_keeps_body(self):
        html = '<!DOCTYPE html><html lang=""></html><head><meta charset="utf-8"></head>'
        html += '<body><h1>Tiêu đề</h1><div class="content-fck">' + "<p>Đoạn văn.</p>" * 20
        html += "</div></body></html>"
        extracted = extract_page(html.encode("utf-8"), None, "baotayninh.vn")
        self.assertEqual(extracted.extractor, "cms:content-fck")
        self.assertIn("Đoạn văn.\n\nĐoạn văn.", extracted.text)

    def test_base64_blob_is_dropped(self):
        blob = "4buxxKjhu7XDveG6uuG7jMSC" * 20
        html = '<div class="article__body">' + "<p>Bài thật.</p>" * 30 + f"<p>{blob}</p></div>"
        extracted = extract_html(f"<html><body><h1>T</h1>{html}</body></html>", "x.vn")
        self.assertNotIn(blob, extracted.text)


@unittest.skipIf(httpx is None, "ingest extra not installed")
class ClassifyTest(unittest.TestCase):
    def test_challenge_pages_are_blocked(self):
        from vibiomir.ingest.fetch import classify_response

        cookie = b'<html><body><script>document.cookie="D1N=1";window.location.reload(true);'
        cookie += b"</script></body></html>"
        self.assertEqual(classify_response(200, cookie), "blocked")
        self.assertEqual(
            classify_response(403, b"<title>Attention Required! | Cloudflare</title>"), "blocked"
        )
        pow_page = b'<h1>Cookies must be enabled</h1><script>const challengeId = "x";</script>'
        self.assertEqual(classify_response(203, pow_page), "blocked")
        self.assertEqual(classify_response(404, b"not here"), "gone")
        self.assertIsNone(classify_response(200, b"<html><script src=recaptcha></script>ok"))
        captcha_url = "https://image.39.net/verify.html?referer=https%3A%2F%2Ffood.39.net%2Fa"
        self.assertEqual(classify_response(200, b"<html>ok</html>", captcha_url), "blocked")
        self.assertIsNone(classify_response(200, b"ok", "https://x.vn/verify-thong-tin.html?a=1"))


class StateTest(unittest.TestCase):
    def test_resume_only_returns_unfinished_work(self):
        with tempfile.TemporaryDirectory() as tmp, CrawlState(Path(tmp) / "s.sqlite") as state:
            links = [(1, "https://a.vn/1"), (2, "https://a.vn/2"), (3, "https://b.cn/3")]
            self.assertEqual(state.seed(links), 3)
            self.assertEqual(state.seed(links), 0)
            state.update([PageUpdate(1, "ok"), PageUpdate(2, "failed", error="timeout")])
            self.assertEqual([i for i, _ in state.iter_todo("a.vn", max_attempts=3)], [2])
            self.assertEqual(list(state.iter_todo("a.vn", max_attempts=1)), [])
            state.exclude([3])
            self.assertEqual(state.domains_todo(max_attempts=3), {"a.vn": 1})
            self.assertEqual(state.reset(["a.vn"], ["ok"]), 1)
            self.assertEqual([i for i, _ in state.iter_todo("a.vn", max_attempts=1)], [1])

    @unittest.skipIf(httpx is None, "ingest extra not installed")
    def test_sink_flushes_slow_crawls_by_time(self):
        from vibiomir.ingest.fetch import ResultSink

        with tempfile.TemporaryDirectory() as tmp, CrawlState(Path(tmp) / "s.sqlite") as state:
            state.seed([(1, "https://a.vn/1"), (2, "https://a.vn/2")])
            sink = ResultSink(state, Path(tmp) / "parts", flush_every=1000, flush_interval=3600)
            sink.add(PageUpdate(1, "ok"), {"id": 1, "text": "x"})
            self.assertEqual(list((Path(tmp) / "parts").glob("*.parquet")), [])
            sink.flush_interval = 0
            sink.add(PageUpdate(2, "ok"), {"id": 2, "text": "y"})
            self.assertEqual(len(list((Path(tmp) / "parts").glob("*.parquet"))), 1)
            self.assertEqual(list(state.iter_todo("a.vn", max_attempts=3)), [])


ARTICLE = """<html><head><meta charset="utf-8"></head><body><h1>Bài {n}</h1>
<div class="article__body">{body}</div></body></html>"""


@unittest.skipIf(httpx is None, "ingest extra not installed")
class CrawlerTest(unittest.TestCase):
    def _config(self, tmp: Path):
        from vibiomir.ingest.domains import DomainPolicy
        from vibiomir.ingest.fetch import CrawlConfig

        return CrawlConfig(
            state_path=tmp / "state.sqlite",
            out_dir=tmp / "parts",
            samples_dir=tmp / "samples",
            extract_workers=0,
            retries=1,
            backoff_base=0.0,
            block_threshold=3,
            cooldown=0.01,
            keep_samples=1,
            log_every=3600,
            default_policy=DomainPolicy(rps=1000, max_rps=1000, concurrency=2),
        )

    def test_crawl_writes_parts_stops_blocked_domain_and_resumes(self):
        from vibiomir.ingest.fetch import Crawler

        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            host, path = request.url.host, request.url.path
            if path == "/robots.txt":
                if host == "ok.vn":
                    return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
                return httpx.Response(404)
            if host == "blocked.vn":
                return httpx.Response(403, text="<title>Attention Required! | Cloudflare</title>")
            if path == "/flaky":
                return httpx.Response(200 if len([c for c in calls if "flaky" in c]) > 1 else 502,
                                      html=ARTICLE.format(n="f", body="<p>" + "chữ " * 80 + "</p>"))
            if path == "/missing":
                return httpx.Response(404)
            if path.endswith(".pptx"):
                return httpx.Response(200, content=b"PK", headers={"content-type": "application/"
                                      "vnd.openxmlformats-officedocument.presentationml.presentation"})
            if path == "/short":
                return httpx.Response(200, html=ARTICLE.format(n="s", body="<p>ngắn</p>"))
            body = "".join(f"<p>Đoạn {i} nói về sức khỏe tim mạch.</p>" for i in range(20))
            return httpx.Response(200, html=ARTICLE.format(n=path, body=body))

        links = [(i, f"https://ok.vn/a{i}") for i in range(1, 6)]
        links += [(6, "https://ok.vn/private/x"), (7, "https://ok.vn/missing"),
                  (8, "https://ok.vn/short"), (9, "https://ok.vn/flaky")]
        links += [(50 + i, f"https://ok.vn/slides{i}.pptx") for i in range(5)]
        links += [(100 + i, f"https://blocked.vn/p{i}") for i in range(20)]

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            crawler = Crawler(self._config(tmp), transport=httpx.MockTransport(handler))
            counts = crawler.run(links)
            self.assertEqual(counts["ok"], 6)
            self.assertEqual(counts["short"], 1)
            self.assertEqual(counts["gone"], 1)
            self.assertEqual(counts["disallowed"], 1)
            self.assertEqual(counts["unsupported"], 5)  # files, not blocks: domain keeps going
            self.assertIsNone(crawler.stats["ok.vn"].stopped)
            self.assertLess(counts["blocked"], 20)  # circuit breaker stopped the domain
            self.assertIsNotNone(crawler.stats["blocked.vn"].stopped)

            rows = pq.read_table(tmp / "parts").to_pylist()
            by_id = {row["id"]: row for row in rows}
            self.assertEqual(sorted(by_id), [1, 2, 3, 4, 5, 8, 9])
            self.assertTrue(by_id[1]["text"].startswith("Bài /a1\n\nĐoạn 0"))
            self.assertEqual(by_id[1]["language"], "vi")
            self.assertEqual(by_id[1]["extractor"], "cms:article-body")
            samples = list((tmp / "samples").rglob("*.html.gz"))
            self.assertTrue(samples)
            self.assertTrue(gzip.decompress(samples[0].read_bytes()).startswith(b"<"))

            calls.clear()
            again = Crawler(self._config(tmp), transport=httpx.MockTransport(handler)).run(links)
            fetched = {c for c in calls if not c.endswith("robots.txt")}
            self.assertFalse(any("ok.vn" in c for c in fetched))  # finished pages not refetched
            self.assertNotIn("ok", again)

    def test_repeated_short_page_is_treated_as_block(self):
        from vibiomir.ingest.fetch import Crawler

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, html="<html><body><p>Đang tải…</p></body></html>")

        links = [(i, f"https://same.vn/p{i}") for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp_name:
            crawler = Crawler(self._config(Path(tmp_name)), transport=httpx.MockTransport(handler))
            counts = crawler.run(links)
        self.assertEqual(counts["short"], 4)
        self.assertEqual(counts["blocked"], 4)

    def test_block_after_successes_pauses_instead_of_stopping(self):
        from vibiomir.ingest.fetch import Crawler

        served = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            served.append(request.url.path)
            if 4 <= len(served) <= 6:
                return httpx.Response(403, text="<title>Access Denied</title>")
            body = "".join(f"<p>Đoạn {i} về dinh dưỡng.</p>" for i in range(20))
            return httpx.Response(200, html=ARTICLE.format(n=request.url.path, body=body))

        links = [(i, f"https://flaky.vn/p{i}") for i in range(15)]
        with tempfile.TemporaryDirectory() as tmp_name:
            crawler = Crawler(self._config(Path(tmp_name)), transport=httpx.MockTransport(handler))
            counts = crawler.run(links)
        stats = crawler.stats["flaky.vn"]
        self.assertIsNone(stats.stopped)
        self.assertEqual(stats.pauses, 1)
        self.assertEqual(counts["blocked"], 3)
        self.assertEqual(counts["ok"], 12)



EFETCH_XML = b"""<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID Version="1">111</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2008</Year><Month>Aug</Month></PubDate></JournalIssue>
</Journal>
<ArticleTitle>Profile of <i>dengue</i> fever.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Dengue is common.</AbstractText>
<AbstractText Label="RESULTS">Fever in <sup>90</sup>% of cases.</AbstractText>
<CopyrightInformation>(c) Someone</CopyrightInformation></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">222</PMID><Article>
<ArticleTitle>Title only.</ArticleTitle></Article>
<OtherAbstract><AbstractText>Traduction.</AbstractText></OtherAbstract>
</MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


class PubMedTest(unittest.TestCase):
    def test_parse_keeps_labels_and_flattens_inline_markup(self):
        from vibiomir.ingest.pubmed import parse_efetch, pmid_from_url

        self.assertEqual(pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/18577999/"), "18577999")
        records = parse_efetch(EFETCH_XML)
        self.assertEqual(
            records["111"]["text"],
            "Profile of dengue fever.\n\nBACKGROUND: Dengue is common.\n\n"
            "RESULTS: Fever in 90% of cases.",
        )
        self.assertEqual(records["111"]["published_at"], "2008-08")
        self.assertEqual(records["222"]["text"], "Title only.")
        self.assertEqual(records["222"]["abstract_chars"], 0)

    @unittest.skipIf(httpx is None, "ingest extra not installed")
    def test_fetch_writes_parts_marks_missing_and_skips_done_batches(self):
        from vibiomir.ingest.pubmed import PubMedConfig, PubMedFetcher

        posts = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(request.content)
            return httpx.Response(200, content=EFETCH_XML)

        links = [(5, "https://pubmed.ncbi.nlm.nih.gov/111/"),
                 (6, "https://pubmed.ncbi.nlm.nih.gov/222/"),
                 (7, "https://pubmed.ncbi.nlm.nih.gov/333/")]
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            with CrawlState(tmp / "s.sqlite") as state:
                state.seed(links)
            config = PubMedConfig(out_dir=tmp / "parts", raw_dir=tmp / "raw",
                                  state_path=tmp / "s.sqlite", batch_size=2)
            counts = PubMedFetcher(config, transport=httpx.MockTransport(handler)).run(links)
            # Both sample abstracts are under 200 chars; 333 is absent from the response.
            self.assertEqual(dict(counts), {"title_only": 2, "missing": 1})
            self.assertEqual(len(posts), 2)
            rows = pq.read_table(tmp / "parts").to_pylist()
            self.assertEqual(sorted(r["id"] for r in rows), [5, 6])
            self.assertEqual(rows[0]["source"], "pubmed_eutils")
            with CrawlState(tmp / "s.sqlite") as state:
                statuses = {d: c for _, d, s_, c in state.summary() for d in [s_]}
            self.assertEqual(statuses, {"short": 2, "gone": 1})
            PubMedFetcher(config, transport=httpx.MockTransport(handler)).run(links)
            self.assertEqual(len(posts), 2)  # finished batches are not fetched again


@unittest.skipIf(httpx is None, "ingest extra not installed")
class BootstrapTest(unittest.TestCase):
    def test_only_official_id_url_pairs_survive(self):
        import pyarrow as pa

        from vibiomir.ingest.bootstrap import convert_dataset

        source = {
            "id": [1, 2, 3, 1], "url": ["https://a.vn/1", "https://a.vn/x", "https://b.cn/3",
                                         "https://a.vn/1"],
            "final_url": [None] * 4, "title": ["T", "T", "", "T"],
            "text": ["T\n\n" + "chữ " * 60, "sai id", "", "dup"],
            "extractor": ["trafilatura"] * 4, "published_at": [None] * 4,
            "raw_sha256": [None] * 4,
        }
        official = {1: "https://a.vn/1", 2: "https://a.vn/2", 3: "https://b.cn/3"}
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pq.write_table(pa.table(source), tmp / "corpus.parquet")
            counts = convert_dataset("lighth0406", tmp / "corpus.parquet", official, tmp / "out")
            rows = pq.read_table(tmp / "out").to_pylist()
        self.assertEqual(dict(counts),
                         {"ok": 1, "id_url_mismatch": 1, "empty_text": 1, "duplicate_id": 1})
        self.assertEqual([(r["id"], r["source"], r["language"]) for r in rows],
                         [(1, "hf_lighth0406", "vi")])


@unittest.skipIf(httpx is None, "ingest extra not installed")
class WaybackTest(unittest.TestCase):
    def test_unfinished_pages_are_fetched_through_the_archive(self):
        from vibiomir.ingest.domains import DomainPolicy
        from vibiomir.ingest.fetch import CrawlConfig, Crawler
        from vibiomir.ingest.wayback import seed_from_crawl, url_template

        requested = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nCrawl-delay: 20\n")
            requested.append(str(request.url))
            if request.url.host != "web.archive.org":
                return httpx.Response(500)
            if request.url.path.endswith("/p2"):
                return httpx.Response(404, text="Wayback Machine has not archived that URL.")
            if request.url.path.endswith("/p3"):
                return httpx.Response(200, text="<title>Just a moment...</title>")
            if "/web/2026id_/" in request.url.path:
                target = request.url.path.replace("/web/2026id_/", "/web/20250101000000id_/")
                return httpx.Response(302, headers={"location": target})
            body = "".join(f"<p>Đoạn {i} về bệnh tiểu đường.</p>" for i in range(20))
            return httpx.Response(200, html=ARTICLE.format(n="w", body=body))

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            with CrawlState(tmp / "crawl.sqlite") as crawl:
                crawl.seed([(i, f"https://blocked.vn/p{i}") for i in range(1, 5)]
                           + [(9, "https://vov.vn/x")])
                crawl.update([PageUpdate(1, "blocked"), PageUpdate(4, "ok")])
                with CrawlState(tmp / "wb.sqlite") as fallback:
                    added = seed_from_crawl(crawl, fallback, skip_domains=["vov.vn"])
            self.assertEqual(added, 3)  # p1 blocked, p2/p3 pending; p4 done, vov skipped
            config = CrawlConfig(
                state_path=tmp / "wb.sqlite", out_dir=tmp / "parts", extract_workers=0,
                retries=0, log_every=3600, url_template=url_template("2026"),
                source="wayback", shared_group="web.archive.org",
                default_policy=DomainPolicy(rps=1000, max_rps=1000, concurrency=2),
            )
            crawler = Crawler(config, transport=httpx.MockTransport(handler))
            counts = crawler.run()
            rows = pq.read_table(tmp / "parts").to_pylist()

        self.assertEqual(dict(counts), {"ok": 1, "gone": 1, "blocked": 1})
        self.assertEqual(list(crawler.groups), ["web.archive.org"])
        # The site's Crawl-delay does not slow the archive down.
        self.assertEqual(crawler.groups["web.archive.org"].limiter.rps, 1000)
        self.assertTrue(all(url.startswith("https://web.archive.org/web/") for url in requested))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], "https://blocked.vn/p1")
        self.assertEqual(rows[0]["source"], "wayback")
        self.assertEqual(rows[0]["final_url"],
                         "https://web.archive.org/web/20250101000000id_/https://blocked.vn/p1")


@unittest.skipIf(httpx is None, "ingest extra not installed")
class MergeTest(unittest.TestCase):
    def test_one_row_per_id_by_quality_source_and_freshness(self):
        import csv

        import pyarrow as pa

        from vibiomir.ingest.fetch import write_part
        from vibiomir.ingest.merge import export_jsonl, iter_corpus, merge_corpus

        long_text = "T\n\n" + "chữ " * 80

        def row(doc_id, source, text, fetched_at, url=None):
            return {"id": doc_id, "url": url or f"https://a.vn/{doc_id}", "title": "T",
                    "text": text, "source": source, "fetched_at": fetched_at,
                    "domain": "wrong", "language": "xx"}

        manifest = {
            "id": [7000000, 10, 20, 30, 40, 50],
            "url": ["https://b.cn/7", *(f"https://a.vn/{i}" for i in (10, 20, 30, 40, 50))],
            "domain": ["b.cn", *["a.vn"] * 5],
            "language": ["zh", *["vi"] * 5],
        }
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pq.write_table(pa.table(manifest), tmp / "manifest.parquet")
            (tmp / "crawl").mkdir()
            (tmp / "wayback").mkdir()
            write_part([row(10, "crawl", long_text + "cũ", "2026-10-25T01:00:00+00:00"),
                        row(20, "crawl", "T\n\nngắn", "2026-10-25T01:00:00+00:00"),
                        row(30, "crawl", long_text, "2026-10-25T01:00:00+00:00")],
                       tmp / "crawl" / "part-a.parquet")
            write_part([row(10, "crawl", long_text + "mới", "2026-10-26T01:00:00+00:00"),
                        row(7000000, "crawl", long_text, "2026-10-26T01:00:00+00:00",
                            url="https://b.cn/other")],
                       tmp / "crawl" / "part-b.parquet")
            write_part([row(20, "wayback", long_text, "2026-10-26T05:00:00+00:00"),
                        row(30, "wayback", long_text + "x", "2026-10-26T05:00:00+00:00"),
                        row(40, "wayback", "T\n\nngắn", "2026-10-26T05:00:00+00:00")],
                       tmp / "wayback" / "part-c.parquet")
            report = merge_corpus(tmp / "manifest.parquet", [tmp / "crawl", tmp / "wayback"],
                                  tmp / "corpus", coverage_path=tmp / "coverage.csv",
                                  bucket_rows=2)
            rows = {r["id"]: r for r in iter_corpus(tmp / "corpus", include_missing=True)}
            with (tmp / "coverage.csv").open(encoding="utf-8") as handle:
                coverage = {r["domain"]: r for r in csv.DictReader(handle)}
            written = export_jsonl(iter_corpus(tmp / "corpus", languages=["vi"]),
                                   tmp / "vi.jsonl")
            documents = __import__("vibiomir.io", fromlist=["load_documents"]).load_documents(
                tmp / "vi.jsonl"
            )
            info = (tmp / "corpus_info.json").read_text(encoding="utf-8")

        self.assertEqual(report.rows, 6)
        self.assertEqual(report.unique_ids, 6)
        self.assertEqual(report.duplicate_rows, 3)
        self.assertEqual(report.url_mismatches, 1)
        self.assertEqual(sorted(rows), [10, 20, 30, 40, 50, 7000000])
        self.assertTrue(rows[10]["text"].endswith("mới"))  # newest crawl wins
        self.assertEqual(rows[20]["source"], "wayback")  # ok beats a short crawl
        self.assertEqual(rows[30]["source"], "crawl")  # crawl beats wayback at equal quality
        self.assertEqual(rows[40]["source"], "wayback")
        self.assertEqual((rows[50]["source"], rows[50]["text"]), ("missing", ""))
        self.assertEqual((rows[10]["domain"], rows[10]["language"]), ("a.vn", "vi"))
        self.assertEqual(rows[7000000]["url"], "https://b.cn/7")  # official url kept
        self.assertEqual(coverage["a.vn"]["ok"], "3")
        self.assertEqual(coverage["a.vn"]["short"], "1")
        self.assertEqual(coverage["a.vn"]["missing"], "1")
        self.assertEqual(coverage["a.vn"]["wayback"], "2")
        self.assertEqual(written, 4)
        self.assertEqual(documents[0].metadata["source"], "crawl")
        self.assertIn('"rows": 6', info)

    def test_home_page_redirects_never_win_and_boilerplate_is_dropped(self):
        import pyarrow as pa

        from vibiomir.ingest.fetch import write_part
        from vibiomir.ingest.merge import iter_corpus, merge_corpus

        hotline = "Gọi HOTLINE để đặt lịch khám."
        ids = list(range(1, 41))
        manifest = {"id": ids, "url": [f"https://a.vn/bai-{i}" for i in ids],
                    "domain": ["a.vn"] * 40, "language": ["vi"] * 40}
        rows = [{"id": i, "url": f"https://a.vn/bai-{i}", "final_url": f"https://a.vn/bai-{i}",
                 "title": f"Bài {i}", "source": "crawl", "fetched_at": "2026-10-25",
                 "text": f"Bài {i}\n\n" + f"Nội dung riêng {i}. " * 20 + f"\n\n{hotline}"}
                for i in ids]
        rows[0] = {**rows[0], "final_url": "https://a.vn/", "text": "Trang chủ\n\n" + "menu " * 80}
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            pq.write_table(pa.table(manifest), tmp / "manifest.parquet")
            (tmp / "crawl").mkdir()
            write_part(rows, tmp / "crawl" / "part-a.parquet")
            report = merge_corpus(tmp / "manifest.parquet", [tmp / "crawl"], tmp / "corpus")
            out = {r["id"]: r for r in iter_corpus(tmp / "corpus", include_missing=True)}
        self.assertEqual(report.home_redirects, 1)
        self.assertEqual(out[1]["source"], "missing")
        self.assertEqual(report.cleaned_docs, 39)
        self.assertNotIn(hotline, out[2]["text"])
        self.assertTrue(out[2]["text"].startswith("Bài 2\n\nNội dung riêng 2."))


class CleanTest(unittest.TestCase):
    def test_noise_boilerplate_and_related_tail_are_removed(self):
        from vibiomir.ingest.clean import BoilerplateModel, clean_text
        from vibiomir.ingest.fetch import redirected_home

        body = "\n\n".join(f"Đoạn {i} nói về bệnh gan." for i in range(10))
        text = f"Tiêu đề\n\n{body}\n\n| |\n\nChia sẻ\n\n{body}\n\nCác tin khác\n\nBài khác 1"
        self.assertEqual(clean_text(text, frozenset({"Chia sẻ"})), f"Tiêu đề\n\n{body}\n\n{body}")
        # A "related" heading early in the page is a section title, not the tail.
        early = f"Tiêu đề\n\nCác tin khác\n\n{body}"
        self.assertEqual(clean_text(early), early)

        model = BoilerplateModel(min_docs=3, min_share=0.5)
        for i in range(4):
            model.observe("a.vn", f"T{i}\n\nriêng {i}\n\nChia sẻ")
        model.observe("b.vn", "T\n\nChia sẻ")
        self.assertEqual(model.boilerplate(), {"a.vn": frozenset({"Chia sẻ"})})

        self.assertTrue(redirected_home("https://a.vn/bai-1/", "https://a.vn"))
        self.assertTrue(redirected_home(
            "https://a.vn/bai-1/", "https://web.archive.org/web/20240101000000id_/https://a.vn/"
        ))
        self.assertFalse(redirected_home("https://a.vn/", "https://a.vn/"))
        self.assertFalse(redirected_home("https://a.vn/bai-1", "https://a.vn/bai-1"))
        self.assertFalse(redirected_home("https://a.vn/bai-1", "https://a.vn/?p=1"))


if __name__ == "__main__":
    unittest.main()
