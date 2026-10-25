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


if __name__ == "__main__":
    unittest.main()
