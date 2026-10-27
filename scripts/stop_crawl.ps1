# Stop ingest jobs started by start_crawl.ps1, including their extraction worker processes.
# Rows already written to parquet parts and the SQLite state are kept; start_crawl.ps1 resumes.
param([string[]]$Jobs = @("crawl", "vov", "pubmed", "wayback"))

$Patterns = @{
    crawl  = "*vibiomir ingest crawl --skip-domain*"
    vov    = "*vibiomir ingest crawl --domain vov.vn*"
    pubmed = "*vibiomir ingest pubmed*"
    wayback = "*vibiomir ingest wayback*"
}

foreach ($job in $Jobs) {
    $pattern = $Patterns[$job]
    if (-not $pattern) { Write-Warning "unknown job $job"; continue }
    $found = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like $pattern }
    foreach ($process in $found) {
        # /T also ends the multiprocessing workers, which otherwise keep running.
        taskkill /PID $process.ProcessId /T /F | Out-Null
        Write-Host "$job stopped (pid $($process.ProcessId))"
    }
    if (-not $found) { Write-Host "$job not running" }
}
