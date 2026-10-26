# Keep Windows from idle-sleeping while any `vibiomir ingest` process is running.
# Uses SetThreadExecutionState (no power-plan change); exits when the last job ends.
# Closing the lid or pressing the power button can still sleep the machine.
param([int]$IntervalSeconds = 60)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

function Get-IngestJobs {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*vibiomir ingest*' }
}

try {
    while (Get-IngestJobs) {
        [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
}
