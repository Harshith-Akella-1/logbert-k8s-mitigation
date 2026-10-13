# Stage 3: record the validation and test fault runs back to back (about 2 h 15 min in total).
# Keep the laptop plugged in with the lid open: standby pauses the injector and corrupts the run.
#   .\infra\fault_runs.ps1
#   .\infra\fault_runs.ps1 -Runs fault_test          # only one of them
param([string[]]$Runs = @('fault_val', 'fault_test'), [int]$Reps = 2, [int]$Duration = 90, [int]$Gap = 120)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (-not $env:LOGMIT_DATA) { $env:LOGMIT_DATA = Join-Path $root 'data' }
$seeds = @{ fault_val = 0; fault_test = 1 }
$minutes = [math]::Ceiling(($Reps * 9 * ($Duration + $Gap) + $Gap) / 60) + 3

# Hold a "system and display required" request so Windows does not enter standby mid-run.
$awake = Start-Process python -ArgumentList '-c', '"import ctypes,time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000003); time.sleep(86400)"' -WindowStyle Hidden -PassThru

foreach ($run in $Runs) {
    $log = Join-Path $env:LOGMIT_DATA "$run.log.txt"
    "$(Get-Date -Format 'HH:mm:ss') $run started" | Tee-Object $log -Append
    $collector = Start-Process python -ArgumentList '-u', '-m', 'logmit.collect', '--run', $run, '--minutes', $minutes `
        -WindowStyle Hidden -RedirectStandardOutput "$env:LOGMIT_DATA\$run.collect.out.txt" `
        -RedirectStandardError "$env:LOGMIT_DATA\$run.collect.err.txt" -PassThru
    cmd /c "python -u -m logmit.faults --run $run --reps $Reps --duration $Duration --gap $Gap --seed $($seeds[$run]) 2>&1" |
        Tee-Object $log -Append
    "$(Get-Date -Format 'HH:mm:ss') $run injector finished (exit $LASTEXITCODE)" | Tee-Object $log -Append
    Start-Sleep -Seconds 90   # keep collecting while the last fault settles
    Stop-Process -Id $collector.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 60
}
Stop-Process -Id $awake.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
"$(Get-Date -Format 'HH:mm:ss') all runs finished" | Tee-Object (Join-Path $env:LOGMIT_DATA 'fault_runs.done.txt')
