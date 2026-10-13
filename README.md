# Proactive microservice failure mitigation with LogBERT on Kubernetes

Logs from Online Boutique on Minikube are parsed with Drain, scored by LogBERT, and a controller
restarts or scales the affected service through the Kubernetes API.

**Rule for the paper:** every number comes from a real run of the commands below. The scripts write
`results/*.json`; the paper is rebuilt from those files only.

## What each file does

| Stage | File | Job |
|---|---|---|
| 1 Environment | `infra/setup.ps1` | Start Minikube, deploy Online Boutique and the prober |
| 2 Log pipeline | `logmit/collect.py` | Poll pod logs into `logs.jsonl` |
| | `logmit/parse.py` | Drain: raw line to template key |
| 3 Data | `logmit/faults.py` | Inject faults, record each interval in `faults.jsonl` |
| | `logmit/windows.py` | 32-key windows per service, stride 8, labelled from `faults.jsonl` |
| 4 Detector | `logmit/logbert.py` | LogBERT training and scoring |
| | `logmit/baselines.py`, `logmit/evaluate.py` | Error-rate rule, Isolation Forest, DeepLog; precision, recall, F1 |
| 5 Controller | `logmit/controller.py`, `logmit/run_loop.py` | Table I policy, 60 s cooldown, 3 actions per 5 min |
| 6 Closed loop | `logmit/metrics.py`, `infra/hpa.yaml` | MTTR, availability, lead time; HPA for the reactive baseline |
| 7 Paper | `build_paper.py` | Rebuild the PDF from `results/*.json` |

A "run" is a named folder under `data/` (for example `data/normal1/`). Data is bulky and this folder
syncs to OneDrive, so point it at another drive first, in every terminal you use:

```powershell
$env:LOGMIT_DATA = "D:\logmit-data"
```

## One-time setup (per laptop)

1. Open PowerShell **as Administrator**, install WSL, then reboot:
   ```powershell
   wsl --install
   ```
2. Install Docker Desktop, start it, and wait for "Engine running":
   ```powershell
   winget install -e --id Docker.DockerDesktop
   ```
   If C: is short on space, move Docker's storage before pulling anything:
   Docker Desktop > Settings > Resources > Advanced > Disk image location.
3. Install kubectl and Minikube (no admin needed) and the Python packages:
   ```powershell
   winget install -e --id Kubernetes.kubectl
   winget install -e --id Kubernetes.minikube
   pip install -r requirements.txt
   ```
   On a laptop with an NVIDIA GPU, install the CUDA build of PyTorch as well. One LogBERT epoch took
   about 5 minutes on CPU and 8 seconds on an RTX 3050:
   ```powershell
   pip install --force-reinstall --no-deps torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
   ```
4. Start the cluster and the application:
   ```powershell
   .\infra\setup.ps1
   kubectl get pods        # every pod should show READY 1/1 and STATUS Running
   ```

`minikube stop` pauses the cluster and frees the RAM; `.\infra\setup.ps1` brings it back.

## Running the experiment

Each numbered block is one terminal. Run all commands from the project folder.

**A. Normal traffic (training data).** Online Boutique ships its own load generator, so the
application is already under load. Collect for as long as you can afford (an hour is a sensible start).

```powershell
python -m logmit.collect --run normal1 --minutes 60
python -m logmit.parse --run normal1 --fit
python -m logmit.logbert --run normal1 --epochs 10
```

**B. Fault runs (validation and test data).** Two terminals at once; stop the collector with Ctrl+C
when the injector finishes. Do it twice, once for `fault_val` and once for `fault_test` (use `--seed 1`).

```powershell
# terminal 1
python -m logmit.collect --run fault_val
# terminal 2
python -m logmit.faults --run fault_val --reps 3
```

```powershell
python -m logmit.parse --run fault_val
python -m logmit.parse --run fault_test
python -m logmit.evaluate --train normal1 --val fault_val --test fault_test
```

This writes `results/detection.json` (Table III and Figs. 2 and 3) and fixes the threshold the live loop uses.

**C. Closed loop.** The same fault schedule twice: once with only Kubernetes' own recovery, once with the controller.

```powershell
# reactive baseline
kubectl apply -f infra\hpa.yaml
python -m logmit.collect --run loop_reactive          # terminal 1
python -m logmit.faults --run loop_reactive --reps 3  # terminal 2
kubectl delete -f infra\hpa.yaml

# proactive
python -m logmit.collect --run loop_proactive          # terminal 1
python -m logmit.run_loop --run loop_proactive         # terminal 2
python -m logmit.faults --run loop_proactive --reps 3  # terminal 3

python -m logmit.metrics --run loop_reactive
python -m logmit.metrics --run loop_proactive
```

Try `python -m logmit.run_loop --run <name> --dry-run` first: it prints what the controller would do without touching the cluster.

**D. Ablations.** Train variants, then evaluate them together:

```powershell
python -m logmit.logbert --run normal1 --no-vhm --name logbert_novhm
python -m logmit.logbert --run normal1 --no-svc --name logbert_nosvc
python -m logmit.logbert --run normal1 --layers 2 --name logbert_2layer
python -m logmit.evaluate --train normal1 --val fault_val --test fault_test --logbert logbert logbert_novhm logbert_nosvc logbert_2layer
```

**E. Paper.** Rebuild the PDF from whatever results exist so far:

```powershell
python build_paper.py        # writes docs\paper.pdf
```

`build_paper.py` contains no numbers of its own. It reads `results/detection.json`,
`results/loop_reactive.json`, `results/loop_proactive.json` and `results/cluster.json`, prints
"pending" for anything missing, and keeps the "provisional" footer until nothing is pending.
The headline recovery numbers cover pod kill and CPU throttling only; the outage fault is reported
in its own paragraph because the reactive baseline cannot recover from it by design.
The discussion sentences only state the measured values. Once the real numbers are in, add your
interpretation by hand in the Results section of `build_paper.py`.

## Where this differs from the original draft (already reflected in `build_paper.py`)

- **Three fault types, not six**, all injected with plain Kubernetes API calls instead of Chaos Mesh:
  repeated pod kill (crash loop), scale-to-zero (dependency outage), in-place CPU limit cut (CPU throttling).
- **One Minikube node** with 4 vCPU and 6 GB, not three nodes and 32 GB. "Reschedule to another node" is not implemented.
- **Log collection polls the Kubernetes API** every 2 s instead of a Fluent Bit DaemonSet.
- **Fault class comes from keyword rules** on the anomalous window, not a trained classifier head.
  "Raise memory request" is not implemented.
- **A window is labelled anomalous only for the faulted service and the services that call it**; the
  60 s after each fault are excluded from scoring.
- **Thresholds (tau, g) are tuned on a validation fault run** and reported on a separate test run, the same way for every method.
- Dataset size, template count, parameter count and all results: take them from the script output.
