Goal: build and evaluate a log-based proactive mitigation loop on Minikube with Online Boutique. Detection uses LogBERT, and the controller acts through the Kubernetes API.
Team context: four IT-3 students, none with Docker or Kubernetes experience. Codex should explain each step as it goes and keep commands copy-pasteable.
Hard rule on results: every number in the paper must come from a real run. If a result is worse than the placeholder, report the real one. Keep the “provisional” notes in the paper until all numbers are replaced.
Priority order: a working end-to-end loop first, then the baselines, then the ablations. The deadline is tight, so a thin vertical slice beats a wide, unfinished build.
Scope cuts allowed: fewer fault types (3 instead of 6), a smaller LogBERT, and fewer repetitions. State each cut in the paper’s limitations.

Build order:
Environment: install Docker, Minikube and kubectl. Deploy Online Boutique and confirm all pods are healthy.
Log pipeline: collect pod logs and parse them with Drain into template keys.
Data generation: run a normal-traffic period with a load generator, then inject faults. Use Chaos Mesh, or plain kubectl kill, delete or stress for speed. Label the windows.
Detector: train LogBERT on normal windows only. Add DeepLog and Isolation Forest as baselines, then compute precision, recall and F1.
Controller: implement the Table I policy with a cooldown and an action budget.
Closed-loop experiment: compare the reactive baseline (probes + HPA) against the proactive controller. Measure MTTR, availability and lead time.
Paper update: regenerate the figures and tables from real outputs and rebuild the PDF.


Decisions:
Fault count: how many fault types you can realistically run in the time you have.
Compute: whether LogBERT trains locally on CPU or on a free Colab GPU. A small model on 1M windows is feasible on CPU but slow.
Task split: one person per stage (infra, data, model, controller) lets four people use Codex in parallel on separate folders.
Keep the paper script: include build_paper.py so it can regenerate the PDF with real numbers.