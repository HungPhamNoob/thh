# Start the corpus ingest jobs as hidden processes that outlive the terminal that started them,
# plus keep_awake.ps1. Every job resumes from data/interim/crawl_state.sqlite and its parts, so
# rerunning this script after a reboot or crash continues where the jobs stopped. A job that is
# already running is not started twice. Logs: data/interim/logs/<job>.log
# The wayback job reads pages the crawl could not fetch from the Wayback Machine (D7).
param(
    [string[]]$Jobs = @("crawl", "vov", "pubmed", "wayback"),
    [string]$Python = (Get-Command python).Source
)

$Root = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Root "data\interim\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null

$Commands = @{
    crawl  = "ingest crawl --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn " +
             "--log-file data/interim/logs/crawl_full.log"
    vov    = "ingest crawl --domain vov.vn --links `"`" --workers 1 " +
             "--log-file data/interim/logs/crawl_vov.log"
    pubmed = "ingest pubmed --log-file data/interim/logs/pubmed.log"
    wayback = "ingest wayback --skip-domain pubmed.ncbi.nlm.nih.gov --skip-domain vov.vn " +
              "--skip-domain zysjonline.com --skip-domain pmc-ecm-healthblog.beta.pharmacity.io " +
              "--concurrency 8 " +
              "--log-file data/interim/logs/wayback.log"
}

function Test-Running([string]$Arguments) {
    $needle = ($Arguments -split " --log-file")[0]
    [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*vibiomir $needle*" })
}

foreach ($job in $Jobs) {
    $arguments = $Commands[$job]
    if (-not $arguments) { Write-Warning "unknown job $job"; continue }
    if (Test-Running $arguments) { Write-Host "$job already running"; continue }
    # A fresh file per start: a killed job's worker processes may still hold the old one open.
    $stdout = Join-Path $Logs ("{0}-{1:yyyyMMdd-HHmmss}.stdout" -f $job, (Get-Date))
    # Win32_Process.Create detaches the job from this console and its process tree.
    $line = "cmd.exe /c set PYTHONPATH=src&& `"$Python`" -m vibiomir $arguments > `"$stdout`" 2>&1"
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $line; CurrentDirectory = $Root; ProcessStartupInformation = $startup
    }
    Write-Host "$job started (pid $($result.ProcessId), return $($result.ReturnValue))"
}

$awake = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like '*keep_awake.ps1*' }
if (-not $awake) {
    Start-Sleep -Seconds 5  # let the jobs appear before keep_awake checks for them
    $script = Join-Path $PSScriptRoot "keep_awake.ps1"
    $line = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $line; CurrentDirectory = $Root
    }
    Write-Host "keep_awake started (pid $($result.ProcessId))"
}
