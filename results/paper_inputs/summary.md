# Paper inputs: what was actually run and measured

Collected 2026-10-14 from files on disk. Every number below comes from a real run and can be traced
to a JSON file in this folder. Closed-loop experiments have **not** been run; section 5 is empty.

| File | Contents |
|---|---|
| `setup.json` | Versions, hardware, cluster, Online Boutique, load generator, prober |
| `cluster.json` | Raw `kubectl version -o json` |
| `data.json` | The three runs: times, line counts, window counts, split, labelling rule, episode lists |
| `templates.json` | The 48 Drain templates |
| `training_curves.json` | Per-epoch training and held-out loss, full model and three ablations |
| `validation_threshold_sweep.json` | LogBERT precision/recall/F1 against tau on the validation run |
| `detection_test.json` | Test-run precision/recall/F1 for all methods, per fault type, ablations |
| `episode_detection_test.json` | Per-episode detection and delay for LogBERT and the rule |

## 1. Setup

| Item | Value |
|---|---|
| Machine | Dell G15 5530, Intel Core i5-13450HX (10 cores / 16 threads), 15.7 GB RAM, Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, driver 596.08 |
| Cluster | Minikube v1.39.0, Docker driver, **1 node, 4 vCPU, 6144 MB**, containerd 2.3.4 |
| Kubernetes | v1.37.0 (kubectl client v1.37.1) |
| Docker | Docker Desktop 4.94.0, engine 29.8.2, VM memory 7.6 GB |
| WSL | 3.0.1.0 during the normal run, 3.0.3.0 during both fault runs (see section 6) |
| Python stack | Python 3.13.7, PyTorch 2.10.0+cu126, CUDA 12.6, drain3 0.9.11, kubernetes client 37.0.1, scikit-learn 1.8.0 |
| Online Boutique | release v0.10.3, commit `cd1cb59f6fce3e2ccf08be7cea743e3784932d73`, manifest applied unchanged, 1 replica per service |
| Load generator | Bundled Locust 2.33.2, **10 users, spawn rate 1/s, constant** (no diurnal pattern), 1–10 s wait between tasks, task weights index 1 / setCurrency 2 / browseProduct 10 / addToCart 2 / viewCart 3 / checkout 1 |
| Prober | One curl pod: GET `/`, `/product/OLJCESPC7Z`, `/cart` in sequence, 0.5 s pause, 3 s timeout; success = 2xx or 3xx |
| Log collection | Kubernetes log API polled every 2 s; `loadgenerator` and `prober` logs are collected but not given to any detector |

## 2. Data

All three runs were recorded on 2026-10-13 (local time, IST).

| Run | Start | End | Minutes | Lines, all pods | Lines, 10 monitored services | Windows |
|---|---|---|---|---|---|---|
| Normal (`normal1`) | 00:45:33 | 01:30:10 | 44.62 | 232,457 | 196,999 | 24,590 |
| Validation (`fault_val`) | 13:12:20 | 14:19:52 | 67.54 | 317,673 | 264,933 | 33,076 |
| Test (`fault_test`) | 14:20:54 | 15:28:26 | 67.54 | 318,967 | 266,494 | 33,272 |

The normal run was planned for 60 minutes and stopped at 44.6 when the session ended. It has no gaps
(largest gap between lines 0.9 s). The two line counts differ because 35,458 lines belong to the load
generator (24,030) and the prober (11,428).

**Templates.** 48, learned by Drain from the normal run only (similarity 0.5, depth 4). Fault runs are
parsed with the templates frozen; unseen lines map to one unknown key (5.65 % of validation lines,
5.14 % of test lines). The normal run is not error-free: it contains one genuine failed checkout
(HTTP 500, rejected card type) and 14 "failed to retrieve ads" warnings, so those messages are known
templates.

**Windows.** 32 keys, stride 8, built separately for each service.

**Split of the normal run** (chronological block, not shuffled; `logmit/windows.py` line 49):

| Part | Time range | Windows | Used for |
|---|---|---|---|
| Training | 00:45:33 – 01:17:40 | 19,635 | LogBERT training |
| Held-out | 01:17:40 – 01:30:10 | 4,918 | LogBERT held-out loss only |
| Dropped | straddle the cut | 37 | nothing |

Validation and test are separate runs, so no window is shared between tuning and reporting.

**Labelling rule, as implemented** (`logmit/windows.py`, `build`):

- A window of service S spanning `[t0, t1]` is **anomalous** if `t0 <= end` and `t1 >= start` for an
  episode whose affected set contains S.
- The **affected set** is the fault target plus every service that calls it, directly or transitively,
  in the static call graph (`logmit/config.py`, `CALLS`). Example: a fault on `redis-cart` marks
  `redis-cart`, `cartservice`, `checkoutservice` and `frontend`.
- A window that is not anomalous but overlaps the **60 s after an episode ends**, for a service in that
  episode's affected set, is **excluded** from tuning and scoring.
- Everything else is **normal**, including windows of unaffected services during a fault.

| Run | Normal | Anomalous | Excluded | Anomalous by fault (cpu_throttle / outage / pod_kill) |
|---|---|---|---|---|
| Validation | 21,557 | 6,640 | 4,879 | 1,095 / 1,678 / 3,867 |
| Test | 21,900 | 6,443 | 4,929 | 1,017 / 1,710 / 3,716 |

**Faults.** 3 types × 3 targets × 2 repetitions = 18 episodes per run, in shuffled order (seed 0 for
validation, 1 for test), 120 s of normal traffic between episodes.

| Fault | How it is injected | Targets | Length |
|---|---|---|---|
| `pod_kill` | Pod force-deleted every 20 s | productcatalogservice, currencyservice, cartservice | 100 s |
| `outage` | Deployment scaled to 0 replicas, then restored | redis-cart, recommendationservice, shippingservice | 90 s |
| `cpu_throttle` | CPU limit of the running pod cut in place to 10m, then restored | frontend, productcatalogservice, currencyservice | 90 s |

The full episode lists with start and end times are in `data.json`. The test list is also in section 4.

## 3. Model and training

| Setting | Value |
|---|---|
| Architecture | Transformer encoder, 4 layers, 4 heads, hidden 256, feed-forward 1024, GELU, dropout 0.1 |
| Parameters | 3,195,955 (no service embedding: 3,193,395; 2 layers: 1,616,435) |
| Input | key + position + service embedding; vocabulary 51 (48 templates + PAD, MASK, UNK) |
| Window / stride | 32 / 8 |
| Mask ratio | 15 % of positions, at least one per window |
| Loss | masked log-key cross-entropy + 0.1 × squared distance to the hypersphere centre; centre recomputed each epoch |
| Optimiser | AdamW, learning rate 1e-3, default weight decay, no schedule |
| Batch / epochs / seed | 256 / 40 / 0 |
| Hardware / time | RTX 3050, 8 s per epoch (about 5 minutes per epoch on CPU) |
| Scoring | each position masked once (8 passes × 4 positions); score = share of positions whose true key is outside the top-g |

**Loss per epoch** (masked log-key loss, training / held-out). All 40 epochs are in `training_curves.json`.

| Epoch | Full model | No VHM | No service embedding | 2 layers |
|---|---|---|---|---|
| 1 | 1.417 / 0.982 | 0.986 / 0.928 | 1.179 / 0.930 | 1.124 / 0.930 |
| 5 | 0.902 / 0.923 | 0.339 / 0.475 | 0.320 / 0.497 | 0.311 / 0.485 |
| 10 | 0.367 / 0.501 | 0.214 / 0.395 | 0.226 / 0.421 | 0.230 / 0.414 |
| 20 | 0.226 / 0.405 | 0.179 / 0.355 | 0.191 / 0.381 | 0.197 / 0.378 |
| 30 | 0.200 / 0.401 | 0.175 / 0.373 | 0.180 / 0.384 | 0.186 / 0.378 |
| 40 | 0.194 / 0.413 | 0.169 / 0.380 | 0.170 / 0.376 | 0.179 / 0.383 |

Two things to note. Held-out loss flattens near 0.38–0.41 from about epoch 20 while training loss keeps
falling to 0.17–0.19, so the models overfit mildly; there is no early stopping and the epoch-40
checkpoint is the one evaluated. The curves come from a re-run with logging added, saved under
`*_rerun` names. The re-run reproduces the original runs' held-out losses exactly (for example 0.3910,
0.3789, 0.4125 for the full model's last three epochs), so they are the curves of the evaluated models.

**Threshold selection** (validation run only, `logmit/evaluate.py`, `tune`): for each g in
{1, 2, 3, 4, 6, 9}, take the tau that maximises window-level F1 on the validation run; keep the best
(g, tau) pair. Excluded windows are left out. Chosen: **g = 3, tau = 0.03125**, meaning a window is
anomalous when at least 2 of its 32 keys fall outside the top 3. The same procedure sets the threshold
of every baseline.

LogBERT on the validation run, g = 3:

| tau | Precision | Recall | F1 |
|---|---|---|---|
| 0.00000 | 0.873 | 0.927 | 0.899 |
| **0.03125** | **0.969** | **0.906** | **0.937** |
| 0.06250 | 0.992 | 0.882 | 0.934 |
| 0.09375 | 0.997 | 0.862 | 0.925 |
| 0.12500 | 0.998 | 0.807 | 0.892 |
| 0.18750 | 0.998 | 0.668 | 0.800 |
| 0.25000 | 0.998 | 0.622 | 0.766 |

**Baselines.**

| Method | Settings |
|---|---|
| Error-rate rule | Score = share of a window's raw lines matching `error\|fail\|exception\|fatal\|panic\|timeout\|timed out\|refused\|unavailable` (case-insensitive). Chosen threshold 0, so one matching line flags the window. |
| Isolation Forest | scikit-learn, 100 trees, other parameters default (`max_samples="auto"`, `contamination="auto"`), random_state 0, on 51-dimensional template-count vectors. Chosen threshold 0.424 on the negated `score_samples`. |
| DeepLog | 2-layer LSTM, embedding 64, hidden 128, history 10 keys, next-key prediction; Adam, learning rate 1e-3, batch 1024, 5 epochs, seed 0. Score = share of the window's 22 predictions outside the top-g. Chosen g = 2, tau = 0.0909. |

Isolation Forest and DeepLog were fitted on all 24,590 normal windows; LogBERT on the first 19,635.
That difference slightly favours the baselines.

## 4. Detection results (test run)

Thresholds were fixed on the validation run. The evaluation script was executed on the test run three
times with the same models and thresholds (one crash from a memory error, two completed runs, the
second only to change the plotted sweep grid); the numbers were identical each time and nothing was
tuned on the test run. The DeepLog model is not saved, so its numbers are from that script run.

**Window level** (21,900 normal, 6,443 anomalous):

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Error-rate rule | 0.997 | 0.815 | 0.897 |
| Isolation Forest | 0.324 | 0.591 | 0.419 |
| DeepLog | 0.943 | 0.855 | 0.897 |
| **LogBERT** | **0.970** | **0.861** | **0.912** |

**F1 by fault type** (that fault's anomalous windows plus all normal windows; 6 episodes per type):

| Method | cpu_throttle | outage | pod_kill |
|---|---|---|---|
| Error-rate rule | 0.487 | 0.969 | 0.938 |
| Isolation Forest | 0.161 | 0.224 | 0.269 |
| DeepLog | 0.577 | 0.885 | 0.905 |
| LogBERT | 0.612 | 0.934 | 0.930 |

**Ablations:**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Full model | 0.970 | 0.861 | 0.912 |
| Without VHM objective | 0.986 | 0.858 | 0.917 |
| Without service embedding | 0.983 | 0.858 | 0.916 |
| Encoder with 2 layers | 0.985 | 0.858 | 0.917 |
| Full model + 2-window persistence rule | 0.984 | 0.843 | 0.908 |

What the numbers do and do not support:

- LogBERT is ahead of DeepLog and the keyword rule by 0.015 F1. This is one run with one seed, so the
  margin has no error bar.
- The keyword rule ties DeepLog. These faults are loud: most produce explicit error lines.
- CPU throttling is the hard case for every method (best F1 0.612).
- No ablation hurts. All three variants score 0.004–0.005 *higher* than the full model, which is within
  what a different seed could produce. The data gives no evidence that VHM, the service embedding or
  the extra two layers help on this dataset.

**Episode level** (LogBERT and the rule; computed offline from window scores, since the live loop has
not been run and no verdict log exists). A verdict is the second of two consecutive anomalous windows
of one affected service. Delay is that window's end time minus the injection time; the live loop would
add up to 2 s of polling. DeepLog is missing because its model was not saved.

| # | Fault | Target | LogBERT verdict (s) | Rule verdict (s) | First failed probe (s) | Probes failed during fault |
|---|---|---|---|---|---|---|
| 1 | pod_kill | currencyservice | 2.45 | 2.45 | 2.48 | 73.3 % |
| 2 | outage | shippingservice | 3.74 | 3.74 | 3.19 | 33.3 % |
| 3 | cpu_throttle | productcatalogservice | 4.52 | 54.13 | none | 0.0 % |
| 4 | cpu_throttle | currencyservice | 4.75 | 12.74 | 11.15 | 19.3 % |
| 5 | pod_kill | productcatalogservice | 3.17 | 1.09 | 3.50 | 29.7 % |
| 6 | pod_kill | productcatalogservice | 3.51 | 5.19 | 3.50 | 26.4 % |
| 7 | cpu_throttle | frontend | 0.56 | 3.56 | 23.53 | 30.6 % |
| 8 | outage | shippingservice | 4.04 | 4.04 | 3.45 | 33.3 % |
| 9 | outage | redis-cart | 0.78 | 0.78 | 0.78 | 100.0 % |
| 10 | cpu_throttle | frontend | 3.42 | 5.02 | 5.83 | 40.6 % |
| 11 | pod_kill | cartservice | 0.47 | 0.47 | 0.46 | 100.0 % |
| 12 | outage | recommendationservice | 6.20 | 6.03 | none | 0.0 % |
| 13 | outage | redis-cart | 2.55 | 6.31 | 3.31 | 100.0 % |
| 14 | cpu_throttle | productcatalogservice | 4.94 | not detected | none | 0.0 % |
| 15 | pod_kill | currencyservice | 2.19 | 2.17 | 2.16 | 66.0 % |
| 16 | cpu_throttle | currencyservice | 4.43 | 9.31 | 7.92 | 85.8 % |
| 17 | pod_kill | cartservice | 0.54 | 0.54 | 0.22 | 100.0 % |
| 18 | outage | recommendationservice | 5.81 | 5.81 | none | 0.0 % |

LogBERT raised a verdict in 18 of 18 episodes (median delay 2.3 s for pod_kill, 3.9 s for outage,
4.5 s for cpu_throttle); the rule in 17 of 18.

Read this table with three cautions:

- **There is almost no early warning.** For pod_kill and outage the verdict arrives at about the same
  moment as the first failed request, sometimes a fraction of a second later. These faults break
  requests immediately, so there is nothing to predict. Only CPU throttling shows a lead: 6.4 s,
  23.0 s, 2.4 s and 3.5 s in the four throttling episodes that had failed requests.
- **18 of 18 overstates the detector.** On normal test windows LogBERT produced 89 false verdict
  windows in 1.13 hours. The `frontend` service emits a window about every 0.3 s, so a verdict within a
  90 s episode can happen by chance. Episodes 3 and 14 (throttled product catalog, zero failed
  requests, rule silent or 54 s late) are the ones most likely to be coincidence.
- **Four episodes had no user impact at all** (3, 12, 14, 18). Recovery time is undefined for them.

## 5. Closed-loop results

**Not run.** No reactive or proactive run has been recorded, so there is no MTTR, availability,
failed-request or action data, and no results file. The controller (`logmit/controller.py`) and live
loop (`logmit/run_loop.py`) have only been exercised in dry-run mode on synthetic logs.

When they are run, `logmit/metrics.py` reports pod_kill and cpu_throttle as the main comparison and
the scale-to-zero outage separately, because the reactive baseline cannot recover from a scale-to-zero
until the fault is withdrawn.

Section 4 already bounds what these runs can show: with verdicts arriving at the moment of failure,
"proactive" mitigation is realistic only for CPU throttling.

## 6. Problems and deviations

**Incidents on 2026-10-13**

| Time | What happened | Effect | Action |
|---|---|---|---|
| 01:30 | Session ended during the normal run | Run is 44.6 min, not 60 | Kept; data has no gaps |
| 08:11–10:24 | LogBERT training on CPU stalled (about 3 s per step; the laptop probably also slept) | No model | Switched to CUDA PyTorch; 8 s per epoch |
| 10:37 | Windows Update installed WSL 3.0.3 and stopped the WSL service, killing Docker and the cluster 2 min into the first validation run | Run lost (`fault_val_aborted`) | Restarted Docker and cluster |
| 10:37 onward | Collector and injector hung silently on the dead cluster | Failure went unnoticed for 2 h | Added 10 s API timeouts; collector aborts after 60 s without a cluster |
| 12:35–13:00 | Laptop on battery entered Modern Standby, stretching one 90 s episode to 6 min 47 s, then rebooted uncleanly | Second validation run lost at 5 of 18 episodes (`fault_val_aborted2`) | Re-ran on mains power with a display-required power request (`infra/fault_runs.ps1`) |
| 15:30 | Evaluation failed twice with a CUDA out-of-memory error while the cluster held most of the RAM | None on results | Stopped Minikube, re-ran |

Neither aborted run is used anywhere. The validation and test runs that are used were recorded back to
back, 13:12–15:28, with every episode at its intended length and no standby events.

**Deviations from the original draft**

| Draft | What was done |
|---|---|
| 6 fault types via Chaos Mesh | 3 fault types via plain Kubernetes API calls |
| 3 nodes, 16 vCPU, 32 GB | 1 node, 4 vCPU, 6 GB |
| 72 h normal traffic, 1.9 M lines | 44.6 min, 196,999 lines |
| 120 users, diurnal load | 10 users, constant load |
| 360 episodes of 5–10 min | 18 episodes per run of 90–100 s, 2 repetitions per fault–target pair |
| 183 templates | 48 templates |
| Fluent Bit DaemonSet | Kubernetes log API polled every 2 s |
| Trained fault-class head | Keyword rules (`logmit/controller.py`, `classify`) |
| Reschedule / cordon / raise memory request | Not implemented: restart one pod, or scale +1 |
| g = 6, tau = 0.20 | g = 3, tau = 0.03125, selected on validation |
| Means over five runs | One run per condition, one seed |
| Any window overlapping a fault is anomalous | Only the target and its callers; 60 s after each fault excluded |
| Inference latency, CPU overhead, false actions per hour per ablation | Not measured |
| Detector runs in the cluster | Collector, detector and controller run on the host |

**Known weaknesses of the current evidence**

- One run and one seed per result; no variance estimate.
- Baselines were fitted on slightly more normal data than LogBERT.
- The CPU throttle is applied to the running pod, so restarting the pod removes it. A restart would
  look more effective here than against a throttle caused by the workload.
- Collector polling loses the last lines of a container that crashes between polls.
- `productcatalogservice` logs nothing under normal traffic; faults on it are visible only through callers.

## 7. Repository

Local git repository only, no remote and no URL. Commit `b50c555` holds the code that produced
`results/detection.json`; `logmit/report.py` and the training-history logging were added after that
commit and are in the following commit together with this folder. The code changed during the day
(parser fixes before the normal run was parsed, collector timeouts before the final fault runs), so the
commit reflects the final code, not a snapshot taken before each run.

All commands are run from the project folder with `LOGMIT_DATA=D:\logmit-data`.

| Result | Command |
|---|---|
| Cluster, `results/cluster.json` | `.\infra\setup.ps1` |
| `normal1` logs | `python -m logmit.collect --run normal1 --minutes 60` (stopped at 44.6 min) |
| Templates, `normal1` keys | `python -m logmit.parse --run normal1 --fit` |
| `logbert.pt` | `python -m logmit.logbert --run normal1 --epochs 40` |
| Ablation models | same with `--no-vhm --name logbert_novhm`, `--no-svc --name logbert_nosvc`, `--layers 2 --name logbert_2layer` |
| `fault_val`, `fault_test` logs and episode lists | `.\infra\fault_runs.ps1` (runs `logmit.collect` and `logmit.faults --reps 2 --duration 90 --gap 120`, seed 0 then 1) |
| Fault-run keys | `python -m logmit.parse --run fault_val`, `python -m logmit.parse --run fault_test` |
| `results/detection.json` | `python -m logmit.evaluate --train normal1 --val fault_val --test fault_test --logbert logbert logbert_novhm logbert_nosvc logbert_2layer` |
| Training curves | the four training commands again with `--name <model>_rerun` |
| Everything else in this folder | `python -m logmit.report --train normal1 --val fault_val --test fault_test` (`setup.json` was written by hand from tool output) |
