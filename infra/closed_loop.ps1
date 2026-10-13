# Stage 6: the closed-loop comparison (about 2 h 25 min). Both runs use the same fault schedule.
#   loop_reactive   Kubernetes only: probes + HPA. The detector runs in dry-run mode to time its verdicts.
#   loop_proactive  The controller acts on the detector's verdicts. No HPA.
# Keep the laptop plugged in with the lid open.
#   .\infra\closed_loop.ps1
param([string[]]$Runs = @('loop_reactive', 'loop_proactive'), [int]$Reps = 2, [int]$Duration = 90, [int]$Gap = 120, [int]$Seed = 2)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (-not $env:LOGMIT_DATA) { $env:LOGMIT_DATA = Join-Path $root 'data' }
$minutes = [math]::Ceiling(($Reps * 9 * ($Duration + $Gap) + $Gap) / 60) + 3
$services = 'adservice','cartservice','checkoutservice','currencyservice','emailservice','frontend','paymentservice','productcatalogservice','recommendationservice','redis-cart','shippingservice'

$awake = Start-Process python -ArgumentList '-c', '"import ctypes,time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000003); time.sleep(86400)"' -WindowStyle Hidden -PassThru

foreach ($run in $Runs) {
    $log = Join-Path $env:LOGMIT_DATA "$run.log.txt"
    # same starting state for both runs: one replica everywhere, HPA only for the reactive run
    cmd /c "kubectl delete -f infra\hpa.yaml --ignore-not-found 2>&1" | Out-Null
    foreach ($s in $services) { cmd /c "kubectl scale deployment $s --replicas=1 2>&1" | Out-Null }
    if ($run -eq 'loop_reactive') { cmd /c "kubectl apply -f infra\hpa.yaml 2>&1" | Tee-Object $log -Append }
    cmd /c "kubectl wait --for=condition=Available deployment --all --timeout=300s 2>&1" | Out-Null
    Start-Sleep -Seconds 120

    "$(Get-Date -Format 'HH:mm:ss') $run started" | Tee-Object $log -Append
    $collector = Start-Process python -ArgumentList '-u', '-m', 'logmit.collect', '--run', $run, '--minutes', $minutes `
        -WindowStyle Hidden -RedirectStandardOutput "$env:LOGMIT_DATA\$run.collect.out.txt" `
        -RedirectStandardError "$env:LOGMIT_DATA\$run.collect.err.txt" -PassThru
    $loopArgs = @('-u', '-m', 'logmit.run_loop', '--run', $run)
    if ($run -eq 'loop_reactive') { $loopArgs += '--dry-run' }
    $detector = Start-Process python -ArgumentList $loopArgs -WindowStyle Hidden `
        -RedirectStandardOutput "$env:LOGMIT_DATA\$run.loop.out.txt" `
        -RedirectStandardError "$env:LOGMIT_DATA\$run.loop.err.txt" -PassThru
    cmd /c "python -u -m logmit.faults --run $run --reps $Reps --duration $Duration --gap $Gap --seed $Seed --reset-replicas 2>&1" |
        Tee-Object $log -Append
    "$(Get-Date -Format 'HH:mm:ss') $run injector finished (exit $LASTEXITCODE)" | Tee-Object $log -Append
    Stop-Process -Id $detector.Id, $collector.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
    cmd /c "kubectl get deployments 2>&1" | Tee-Object $log -Append
    cmd /c "python -m logmit.metrics --run $run 2>&1" | Tee-Object $log -Append
}
cmd /c "kubectl delete -f infra\hpa.yaml --ignore-not-found 2>&1" | Out-Null
Stop-Process -Id $awake.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
"$(Get-Date -Format 'HH:mm:ss') all runs finished" | Tee-Object (Join-Path $env:LOGMIT_DATA 'closed_loop.done.txt')
