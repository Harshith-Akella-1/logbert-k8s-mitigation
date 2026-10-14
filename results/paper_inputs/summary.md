# Paper inputs: what was actually run and measured

Collected 2026-10-14 from files on disk. Every number below comes from a real run and can be traced
to a JSON file in this folder. Section 5 (closed loop) was added after the runs on the same day.

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

Added 2026-10-14. Raw files: `loop_reactive.json`, `loop_proactive.json` (copies of `results/`).

**Protocol.** One new schedule (seed 2, 18 episodes, 90–100 s, 120 s gaps), run twice:

- `loop_reactive` (01:47–02:53): plain Kubernetes, with the HPA from `infra/hpa.yaml` (70 % CPU on
  frontend, productcatalogservice, currencyservice). The detector ran in observe-only (dry-run) mode
  to record verdict times.
- `loop_proactive` (09:25–10:33): the controller acts on verdicts; no HPA.
- In both runs the injector sets every Deployment back to 1 replica 60 s after each episode.

**Pod kill and CPU throttling (12 episodes):**

| Metric | Reactive | Controller |
|---|---|---|
| MTTR, mean over episodes with failed probes (s) | 97.37 (4 episodes) | 32.20 (10 episodes) |
| MTTR, the 4 episodes where both failed (s) | 97.4 | 41.8 |
| Availability (%) | 92.07 | 92.43 |
| Failed probes during faults (%) | 12.55 | 11.01 |
| Episodes not recovered before fault end | 0 | 0 |
| Automatic actions, whole run | 0 | 36 |
| Unnecessary actions per hour | 0 | 4.54 |
| Median lead over first failed probe (s) | 0.08 (observe-only) | 0.00 |
| Availability, whole run including outages (%) | 91.14 | 90.47 |

**Outage (6 episodes, reported separately):** MTTR 73.39 s reactive against 56.07 s with the
controller; failed probes during faults 11.94 % against 18.52 %.

**Per episode** (percentage of probes failed during the fault / MTTR in s / lead in s):

| # | Fault | Target | Reactive | Controller |
|---|---|---|---|---|
| 1 | outage | shippingservice | 28.4 / 125.6 / −0.6 | 3.9 / 39.9 / −0.6 |
| 2 | outage | redis-cart | 9.1 / 89.0 / – | 55.2 / 104.8 / 0.0 |
| 3 | cpu_throttle | productcatalogservice | 0.0 / – / – | 0.0 / – / – |
| 4 | cpu_throttle | currencyservice | 2.1 / 45.1 / 38.0 | 4.2 / 23.4 / −2.6 |
| 5 | cpu_throttle | frontend | 5.8 / 82.0 / 0.2 | 3.7 / 29.2 / 2.4 |
| 6 | pod_kill | productcatalogservice | 0.0 / – / – | 2.7 / 20.3 / 0.0 |
| 7 | cpu_throttle | currencyservice | 0.0 / – / – | 4.5 / 85.2 / 5.6 |
| 8 | outage | redis-cart | 9.0 / 88.9 / 0.0 | 21.5 / 53.3 / −0.6 |
| 9 | pod_kill | productcatalogservice | 0.0 / – / – | 2.3 / 12.5 / 1.6 |
| 10 | outage | recommendationservice | 1.1 / 17.1 / 2.9 | 2.0 / 8.3 / 5.4 |
| 11 | pod_kill | currencyservice | 0.0 / – / – | 40.7 / 9.7 / 0.0 |
| 12 | pod_kill | cartservice | 71.4 / 137.5 / −1.2 | 53.8 / 86.5 / 0.0 |
| 13 | cpu_throttle | frontend | 0.0 / – / – | 2.5 / 12.1 / −0.5 |
| 14 | outage | recommendationservice | 0.4 / 0.1 / 2.7 | 2.1 / 8.4 / 5.1 |
| 15 | outage | shippingservice | 23.6 / 119.7 / −0.5 | 26.4 / 121.6 / 1.3 |
| 16 | cpu_throttle | productcatalogservice | 0.0 / – / – | 0.0 / – / – |
| 17 | pod_kill | cartservice | 71.2 / 124.9 / 0.0 | 15.0 / 28.0 / −0.5 |
| 18 | pod_kill | currencyservice | 0.0 / – / – | 2.8 / 15.2 / −0.6 |

**Controller actions (36):** restart frontend 11, scale frontend 7, scale redis-cart 6, scale
recommendationservice 5, restart cartservice 2, scale checkoutservice 2, restart shippingservice 1,
scale cartservice 1, scale currencyservice 1. Of 2,337 verdict records, 1,880 were "alert:persistent",
300 "alert:budget" and 121 "skip:cooldown".

**Reading the comparison**

- MTTR means are over different episodes. Only the 4 paired episodes compare like with like; the
  controller is faster in all 4.
- 6 episodes failed only with the controller. A likely cause is that the reactive run's HPA scaled
  the frontend to 4–5 replicas from episode 5 onward and kept them despite the reset (4–5 frontend
  pods log in every later episode). The controller run had no HPA. The test run, with neither,
  lost 66–73 % of probes to the currency-service kill that cost the reactive run nothing.
- None of those 6 failures was started by the controller: in every one the first failed probe came
  before any frontend restart. In 4 (episodes 6, 9, 13, 18) the controller restarted the frontend
  0.6–1.8 s after the failures began, in reaction to them; each burst ended within 11–19 s. In
  episode 7 the only restart came 81.6 s after the first failure, and in episode 11 there was none.
  In the pod-kill episodes, later kills caused no further failures once the controller had scaled
  the killed service +1. The gap is the baseline's extra frontend replicas, not controller harm.
- No early warning: median lead 0.1 s observe-only and 0.0 s acting. The exception is CPU throttling
  in observe-only mode (leads 38.0 s and 0.2 s in its two failing episodes).

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

**Incidents on 2026-10-14 (closed loop)**

| Time | What happened | Effect | Action |
|---|---|---|---|
| 01:51 | Detector crashed on the first verdict from a non-frontend service (missing entry in the controller's dependency table; my bug) | Reactive run: no verdict times for 01:51:41–01:56:49 (about episode 2); probe measurements unaffected | Fixed (commit `ea18d02`), restarted the observe-only detector |
| 03:14 | Charger unplugged, laptop entered standby until 06:56 | Proactive run lost after 5 of 18 episodes (`loop_proactive_aborted`) | Re-ran 09:25–10:33 on mains power |

Before the closed-loop runs the controller's dependency naming was changed to read only error
messages; it had matched the `/cart` request path and named `cartservice` for nearly every fault.
This was developed and checked on the validation run.

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

| `loop_reactive.json`, `loop_proactive.json` | `.\infra\closed_loop.ps1` (reactive run), then `.\infra\closed_loop.ps1 -Runs loop_proactive` after the interrupted first attempt; each ends with `python -m logmit.metrics --run <run>` |

## 8. Unattended session, 2026-10-15

New result files: `results/seeds.json`, `results/hdfs.json`, `results/bgl.json`,
`results/figures/fig6_threshold.{pdf,png}`. The paper and `.tex` files were not edited.

### T0. Fig. 6 redrawn

`python results/figures/make_fig6.py` reads the 11 test-run sweep points and the validation-chosen
tau = 1/32 from `results/detection.json` and writes a vector PDF and a 300-dpi PNG. No data changed.

### T2. Seed repeats on the testbed (5 seeds)

Same settings as the paper (LogBERT 4 layers / 256 hidden / 40 epochs; DeepLog as in section 3).
For each seed: train on `normal1`, choose (g, tau) on `fault_val` with the usual procedure, report
`fault_test` once. Command: `python -m logmit.seeds --train normal1 --val fault_val --test fault_test`.

| Model | Seed | g | tau | Precision | Recall | F1 | F1 cpu_throttle | F1 outage | F1 pod_kill |
|---|---|---|---|---|---|---|---|---|---|
| LogBERT | 0 | 3 | 0.0312 | 0.970 | 0.861 | 0.912 | 0.612 | 0.934 | 0.930 |
| LogBERT | 1 | 3 | 0.0312 | 0.981 | 0.852 | 0.912 | 0.594 | 0.948 | 0.938 |
| LogBERT | 2 | 3 | 0.0312 | 0.971 | 0.858 | 0.911 | 0.589 | 0.936 | 0.932 |
| LogBERT | 3 | 3 | 0.0312 | 0.981 | 0.866 | 0.920 | 0.653 | 0.950 | 0.939 |
| LogBERT | 4 | 3 | 0.0312 | 0.978 | 0.854 | 0.912 | 0.586 | 0.946 | 0.937 |
| DeepLog | 0 | 2 | 0.0909 | 0.943 | 0.855 | 0.897 | 0.577 | 0.885 | 0.905 |
| DeepLog | 1 | 2 | 0.0909 | 0.959 | 0.861 | 0.907 | 0.631 | 0.907 | 0.918 |
| DeepLog | 2 | 6 | 0.0000 | 0.991 | 0.820 | 0.897 | 0.436 | 0.964 | 0.943 |
| DeepLog | 3 | 3 | 0.0455 | 0.960 | 0.841 | 0.896 | 0.519 | 0.916 | 0.918 |
| DeepLog | 4 | 2 | 0.0909 | 0.959 | 0.859 | 0.906 | 0.629 | 0.904 | 0.917 |

Mean ± sample standard deviation over the 5 seeds:

| Model | Precision | Recall | F1 | F1 cpu_throttle | F1 outage | F1 pod_kill |
|---|---|---|---|---|---|---|
| LogBERT | 0.976 ± 0.005 | 0.858 ± 0.006 | 0.913 ± 0.004 | 0.607 ± 0.028 | 0.943 ± 0.008 | 0.935 ± 0.004 |
| DeepLog | 0.962 ± 0.018 | 0.847 ± 0.017 | 0.901 ± 0.005 | 0.558 ± 0.082 | 0.915 ± 0.029 | 0.920 ± 0.014 |

- Seed 0 reproduces the original model exactly: the same loss curve, and test F1 0.912 with g = 3,
  tau = 1/32. GPU training here is deterministic for a fixed seed.
- LogBERT is ahead of DeepLog on F1 in every seed, by 0.005–0.024; the mean gap is 0.012.
- DeepLog's chosen (g, tau) changes across seeds; LogBERT's never does.
- Deviation: training ran while the GPU was shared with the HDFS job, so the training times in
  `seeds.json` (668–1003 s for LogBERT) are not clean timings.

### T1. Loghub HDFS_v1

Command: `python -m logmit.bench --dataset hdfs --epochs 10` (module `logmit/bench.py`).

| Item | Value |
|---|---|
| Sessions (block_id) / anomalous | 575,061 / 16,838 |
| Templates | 29 (Loghub's preprocessed event traces) |
| Order | position of the block's first line in `HDFS.log` |
| Train (first 60 %) | 345,036 sessions; 333,878 normal used, 11,158 anomalous dropped |
| Validation (next 20 %) | 115,012 sessions, 4,001 anomalous |
| Test (last 20 %) | 115,013 sessions, 1,679 anomalous |

| Method | Precision | Recall | F1 | Validation F1 | g | tau | Train time (s) |
|---|---|---|---|---|---|---|---|
| Keyword rule | 1.000 | 0.489 | 0.657 | 0.746 | – | 0.0000 | 0 |
| Isolation Forest | 0.022 | 0.917 | 0.043 | 0.330 | – | 0.3527 | 1.9 |
| DeepLog | 0.685 | 0.645 | 0.664 | 0.708 | 4 | 0.0270 | 1,462.6 |
| LogBERT | 0.864 | 0.721 | 0.786 | 0.933 | 4 | 0.0312 | 3,496.6 |

How the methods see a session:

- **LogBERT:** each session is cut into windows of 32 keys (the last one padded, padding excluded
  from attention, masking and scoring); the session score is the highest window score. Same
  architecture as the testbed (4 layers, 4 heads, hidden 256; 3,183,648 parameters, no service
  embedding), 10 epochs on 338,477 training windows.
- **DeepLog:** each key is predicted from the previous 10, padded at the session start; session
  score = share of the session's keys outside the top-g. 5 epochs.
- **Isolation Forest:** template-count vector per session.
- **Keyword rule:** share of a session's events whose template contains a keyword.

Findings:

- **The chronological split has strong drift.** Normal test-period sessions are shorter (median
  13 events against 19 in training) and contain far fewer delete events (E21/E23: 1.19 per session
  against about 2.8), because the log ends before many of those blocks are deleted. Count-based
  Isolation Forest flags most of them (test F1 0.043). Every method scores lower on test than on
  validation (LogBERT 0.933 → 0.786, Isolation Forest 0.330 → 0.043); LogBERT stays best.
- **Not comparable with published numbers.** HDFS results in the literature (F1 around 0.9 or
  higher) mostly use random splits.

Deviations:

- The Loghub-provided templates were used instead of re-running Drain.
- The keyword rule reads template text, not raw lines.
- 10 LogBERT epochs instead of 40, for time.
- The GPU was shared with T2, so the training times are inflated.
- The first launch crashed on a string-formatting bug before any method ran; it was restarted.

### T3. Loghub BGL

Command: `python -m logmit.bench --dataset bgl --epochs 10`.

| Item | Value |
|---|---|
| Lines used | 4,747,936 of 4,747,963 (the last 27 do not fill a window) |
| Sequences | windows of 32 consecutive lines, non-overlapping (stride 32), in file order: 148,373 |
| Anomalous windows | 13,128 (a window is anomalous if any line has a non-'-' label) |
| Parser | drain3, similarity 0.5, depth 4, testbed masks, fitted on the first 60 % of lines only: 324 templates |
| Unknown keys after the cut | 28.5 % of lines match no training-period template |
| Train | 89,023 windows; 80,813 normal used, 8,210 anomalous dropped |
| Validation | 29,675 windows, 2,923 anomalous |
| Test | 29,675 windows, 1,995 anomalous |

| Method | Precision | Recall | F1 | Validation F1 | g | tau | Train time (s) |
|---|---|---|---|---|---|---|---|
| Keyword rule | 0.170 | 0.599 | 0.265 | 0.342 | – | 0.6875 | 0 |
| Isolation Forest | 0.041 | 0.177 | 0.067 | 0.105 | – | 0.3055 | 0.4 |
| DeepLog | 0.122 | 0.827 | 0.213 | 0.403 | 9 | 0.9688 | 141.2 |
| LogBERT | 0.083 | 0.985 | 0.152 | 0.309 | 9 | 0.0625 | 301.7 |

Findings:

- **All four methods fail under this protocol**, and LogBERT is last. With a chronological split
  and templates learned only from the training period, 28.5 % of later lines are unknown to every
  model, so the sequence models flag most windows (LogBERT recall 0.985 at precision 0.083).
- **The keyword rule is weak too.** BGL normal lines often contain words like "error" (for example
  "instruction cache parity error corrected" is labelled normal).
- **Not comparable with published numbers.** Those mostly use random splits, templates from the
  full log, or time windows.

Deviations:

- The first attempt (02:59) used drain3's `match()`, which accepts only a perfect template match,
  and left 61.4 % of post-cut lines unknown. It was stopped after the keyword rule and Isolation
  Forest had been scored (rule F1 0.265, unchanged afterwards; IF F1 0.061) and restarted with
  Drain's tree search at the training similarity threshold. That decision was taken from the
  unknown-key rate, not from any detector's test score; the log of the first attempt is
  `D:\logmit-data\bgl_attempt1_strict_match.out.txt`.
- 10 LogBERT epochs instead of 40.
- The testbed itself still uses strict matching. Its fault runs had 5.1–5.7 % unknown lines, mostly
  genuinely new error messages, so the testbed results are not affected the same way.

### T4. Closed loop with the controller and the HPA together (`loop_proactive_hpa`)

Command: `.\infra\closed_loop.ps1 -Runs loop_proactive_hpa`. Run 03:18–04:24 on mains power, with no
standby events. Same seed-2 schedule as `loop_reactive` and `loop_proactive`, the same HPA as the
reactive run, the controller acting, replicas reset to 1 after each episode as before. New in this
run: the replica count of every Deployment is logged at each episode start (`replicas_at_start` in
`D:\logmit-data\loop_proactive_hpa\faults.jsonl`).

**Pod kill and CPU throttling (12 episodes), all three runs:**

| Metric | Reactive (HPA) | Controller | Controller + HPA |
|---|---|---|---|
| MTTR, mean over episodes with failed probes (s) | 97.37 (4) | 32.20 (10) | 53.61 (4) |
| Availability (%) | 92.07 | 92.43 | 93.33 |
| Failed probes during faults (%) | 12.55 | 11.01 | 9.57 |
| Episodes not recovered before fault end | 0 | 0 | 0 |
| Automatic actions, whole run | 0 | 36 | 37 |
| Unnecessary actions per hour | 0 | 4.54 | 8.17 |
| Median lead over first failed probe (s) | 0.08 (observe-only) | 0.00 | 4.81 |
| Availability, whole run including outages (%) | 91.14 | 90.47 | 92.09 |

**Outage (6 episodes):** MTTR 73.4 s reactive, 56.1 s controller, 54.0 s controller + HPA; failed
probes during faults 11.9 %, 18.5 % and 16.9 %.

**Per episode** (percentage of probes failed during the fault / MTTR in s; replicas above 1 at the
start of the episode in the controller + HPA run):

| # | Fault | Target | Reactive | Controller | Controller + HPA | Replicas > 1 at start (controller + HPA) |
|---|---|---|---|---|---|---|
| 1 | outage | shippingservice | 28.4 / 125.6 | 3.9 / 39.9 | 14.3 / 84.3 | frontend 2 |
| 2 | outage | redis-cart | 9.1 / 89.0 | 55.2 / 104.8 | 49.4 / 111.6 | frontend 2 |
| 3 | cpu_throttle | productcatalogservice | 0.0 / – | 0.0 / – | 0.0 / – | frontend 3 |
| 4 | cpu_throttle | currencyservice | 2.1 / 45.1 | 4.2 / 23.4 | 1.9 / 28.9 | frontend 3, productcatalog 2 |
| 5 | cpu_throttle | frontend | 5.8 / 82.0 | 3.7 / 29.2 | 0.0 / – | currency 4, frontend 4, productcatalog 2 |
| 6 | pod_kill | productcatalogservice | 0.0 / – | 2.7 / 20.3 | 0.0 / – | currency 5, frontend 4, productcatalog 2 |
| 7 | cpu_throttle | currencyservice | 0.0 / – | 4.5 / 85.2 | 0.0 / – | currency 5, frontend 4, payment 2, productcatalog 2 |
| 8 | outage | redis-cart | 9.0 / 88.9 | 21.5 / 53.3 | 31.0 / 85.6 | currency 5, frontend 4, productcatalog 2 |
| 9 | pod_kill | productcatalogservice | 0.0 / – | 2.3 / 12.5 | 0.0 / – | same |
| 10 | outage | recommendationservice | 1.1 / 17.1 | 2.0 / 8.3 | 0.4 / 0.5 | same |
| 11 | pod_kill | currencyservice | 0.0 / – | 40.7 / 9.7 | 0.1 / 0.0 | same |
| 12 | pod_kill | cartservice | 71.4 / 137.5 | 53.8 / 86.5 | 44.4 / 74.7 | same |
| 13 | cpu_throttle | frontend | 0.0 / – | 2.5 / 12.1 | 0.0 / – | same |
| 14 | outage | recommendationservice | 0.4 / 0.1 | 2.1 / 8.4 | 1.1 / 16.3 | currency 4, frontend 4, productcatalog 2 |
| 15 | outage | shippingservice | 23.6 / 119.7 | 26.4 / 121.6 | 5.2 / 25.9 | same |
| 16 | cpu_throttle | productcatalogservice | 0.0 / – | 0.0 / – | 0.0 / – | same |
| 17 | pod_kill | cartservice | 71.2 / 124.9 | 15.0 / 28.0 | 68.4 / 110.9 | currency 5, frontend 4, productcatalog 2 |
| 18 | pod_kill | currencyservice | 0.0 / – | 2.8 / 15.2 | 0.0 / – | same |

**Controller actions (37):** scale frontend 12, restart frontend 6, scale currencyservice 4, scale
recommendationservice 4, scale checkoutservice 3, restart cartservice 2, scale paymentservice 2,
scale redis-cart 2, restart shippingservice 1, scale cartservice 1.

**Reading it**

- **The replica log settles the earlier question.** The reset to 1 replica does not hold while an
  HPA is present: the frontend runs 2–4 replicas and currencyservice 4–5 from episode 5 onward.
  Episode 1 already starts with 2 frontend replicas, because the controller scaled the frontend
  during the 2-minute warm-up before any fault, which is a false action.
- **With the same replica help, the controller adds a little.** Failed episodes drop to 4 (as in
  the reactive run). Availability is 93.33 % against 92.07 %, and failed probes during faults
  9.57 % against 12.55 %. In the three episodes where both runs failed (4, 12, 17), recovery took
  28.9 / 74.7 / 110.9 s against 45.1 / 137.5 / 124.9 s.
- **The cost is false actions.** 8.2 unnecessary actions per hour, higher than without the HPA,
  likely because scaled-up services add log streams that can raise false verdicts (not verified).
- **No early warning.** The median lead of 4.81 s mixes the single CPU-throttle episode with
  failed probes (lead 9.5 s) with pod kills (median 0.12 s). Outage episodes, outside this table,
  have a median lead of 1.36 s.
- **Still one run per condition.** The availability differences are about 1 percentage point and
  have no error bars.
