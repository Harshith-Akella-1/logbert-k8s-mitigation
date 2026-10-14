# Log-based proactive failure mitigation on Kubernetes (LogBERT + Online Boutique)

Code and measured results for a closed-loop experiment: container logs from the Online Boutique demo
application are parsed with Drain, scored per service with LogBERT, and a rule-based controller
restarts or scales Deployments through the Kubernetes API. The repository also contains the
detection baselines (keyword rule, Isolation Forest, DeepLog) and evaluations on the public Loghub
HDFS and BGL datasets.

Every number in `results/` comes from a real run on the setup described below.
`results/paper_inputs/summary.md` explains each run, its settings and every deviation from the
original plan.

## What is in here

| Path | Contents |
|---|---|
| `logmit/collect.py` | Polls every pod's logs through the Kubernetes API |
| `logmit/parse.py` | Drain template extraction (drain3) |
| `logmit/windows.py` | Per-service windows of 32 keys, stride 8, labelled from the fault log |
| `logmit/logbert.py` | LogBERT: masked log-key prediction + hypersphere loss, top-g scoring |
| `logmit/baselines.py`, `logmit/evaluate.py` | Isolation Forest, DeepLog, keyword rule; thresholds chosen on validation |
| `logmit/faults.py` | Fault injection: pod kill, scale-to-zero outage, in-place CPU-limit cut |
| `logmit/controller.py`, `logmit/run_loop.py` | Mitigation controller and live detection loop |
| `logmit/metrics.py` | MTTR, availability and lead time from the prober |
| `logmit/report.py` | Collects data, curves and episode-level detection into `results/paper_inputs/` |
| `logmit/seeds.py` | Seed repeats |
| `logmit/bench.py` | Loghub HDFS / BGL evaluation with the same protocol |
| `infra/` | Cluster setup, prober, HPA, fault-run and closed-loop scripts (PowerShell) |
| `results/` | All measured results (JSON) and figures |

## Tested setup

- **Laptop:** Windows 11, Intel Core i5-13450HX, 16 GB RAM, NVIDIA RTX 3050 6 GB.
- **Cluster:** Docker Desktop 4.94 (WSL 2), Minikube 1.39.0 with Kubernetes v1.37.0, one node with
  4 vCPU and 6 GB.
- **Application:** Online Boutique v0.10.3, release manifest unchanged.
- **Software:** Python 3.13, PyTorch 2.10.0 with CUDA 12.6.

Full versions are in `results/paper_inputs/setup.json`.

## Reproduce

All commands are PowerShell, run from the repository root. Raw logs are not included: each run's
`logs.jsonl` is 70–100 MB, about 560 MB for the six runs behind the results. The commands below
regenerate them. Set a data folder
on a disk with space first:

```powershell
$env:LOGMIT_DATA = "D:\logmit-data"
pip install -r requirements.txt
# GPU (optional, about 40x faster training):
pip install --force-reinstall --no-deps torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

**1. Cluster and application** (needs Docker Desktop running)

```powershell
.\infra\setup.ps1
kubectl get pods        # all pods 1/1 Running
```

**2. Normal traffic, templates, LogBERT** (the published run collected 44.6 minutes)

```powershell
python -m logmit.collect --run normal1 --minutes 45
python -m logmit.parse --run normal1 --fit
python -m logmit.logbert --run normal1 --epochs 40
python -m logmit.logbert --run normal1 --epochs 40 --no-vhm --name logbert_novhm
python -m logmit.logbert --run normal1 --epochs 40 --no-svc --name logbert_nosvc
python -m logmit.logbert --run normal1 --epochs 40 --layers 2 --name logbert_2layer
```

**3. Validation and test fault runs** (about 2 h 15 min; keep the laptop on mains with the lid open)

```powershell
.\infra\fault_runs.ps1
python -m logmit.parse --run fault_val
python -m logmit.parse --run fault_test
```

**4. Detection results**

```powershell
python -m logmit.evaluate --train normal1 --val fault_val --test fault_test --logbert logbert logbert_novhm logbert_nosvc logbert_2layer
python -m logmit.report --train normal1 --val fault_val --test fault_test
python -m logmit.seeds --train normal1 --val fault_val --test fault_test --seeds 0 1 2 3 4
```

**5. Closed loop** (about 70 minutes per run; all runs use fault schedule seed 2)

```powershell
.\infra\closed_loop.ps1 -Runs loop_reactive        # Kubernetes probes + HPA, detector observing only
.\infra\closed_loop.ps1 -Runs loop_proactive       # controller acting, no HPA
.\infra\closed_loop.ps1 -Runs loop_proactive_hpa   # controller acting, with the same HPA
```

**6. Public benchmarks** (Loghub, https://zenodo.org/records/8196385)

```powershell
# unpack HDFS_v1.zip into $env:LOGMIT_DATA\benchmarks\HDFS_v1 and BGL.zip into ...\benchmarks\BGL
python -m logmit.bench --dataset hdfs --epochs 10
python -m logmit.bench --dataset bgl --epochs 10
# supervised LogRobust-style baseline on HDFS (needs transformers and bert-base-uncased)
pip install transformers==4.46.3
python -m logmit.bench_extra --epochs 5
```

## Results at a glance

| Experiment | File |
|---|---|
| Testbed detection (test run), ablations | `results/detection.json` |
| Five-seed repeats | `results/seeds.json` |
| Closed loop | `results/loop_reactive.json`, `results/loop_proactive.json`, `results/loop_proactive_hpa.json` |
| Loghub HDFS / BGL | `results/hdfs.json`, `results/bgl.json` |
| Supervised LogRobust-style baseline on HDFS | `results/hdfs_extra.json` |
| Everything explained | `results/paper_inputs/summary.md` |

## Known limitations

- One demo application on a single-node cluster on a laptop.
- Three fault types.
- One run per closed-loop condition.
- The controller's fault classes come from keyword rules.

Long runs on a laptop are fragile: standby on battery, Windows Update restarting WSL, and low memory
all broke runs during this work (see `summary.md`, section 6).

## License

MIT, see `LICENSE`.
