Cross-architecture figures for GPU recovery paper

Figure 6:
- RTX 4060: existing 100 ms L2 duration experiment, 3 runs.
- Tesla T4: 32-block L2 calibration runs, 3 runs.
- Values are normalized to each run's own baseline median.
- Open points are individual runs; filled point is the median across runs.
- Panel (b) reports recovery P95, not a significance test.

Figure 7:
- Extended L2 experiments, 10 control + 10 interference trials per architecture.
- Each point is one trial's fraction of recovery probes above 1.2x baseline.
- No significance annotation is included.

Interpretation:
These figures are intended as a cross-architecture comparison, not a claim that the two platforms experienced identical absolute interference strength. The T4 and RTX 4060 use different GPU architectures and experimental configurations. The comparison is therefore framed using normalized victim latency and within-platform trial summaries.
