# Learned shortcut time-size experiment

## Question

If a feed-forward model is allowed to learn direct shortcuts, how much capacity
does it need to answer increasingly deep pointer queries, and does fitting those
queries amount to reusable computation?

## Protocol

- Ten matched 50-node cyclic pointer tables were used (`seed=42,...,51`).
- A table-specific, one-hidden-layer ReLU MLP received a one-hot start node and
  a six-bit depth code.
- For each horizon \(H\), one MLP had to answer every query at every depth
  \(1,\ldots,H\). It was directly supervised on all 50 start nodes at all of
  those depths, so shortcut learning was unrestricted.
- The reported frontier is the smallest registered hidden width for which at
  least two of three restarts reached at least 95% accuracy at every depth in
  the prefix.
- The matched AC was trained on the one-hop table once and reused the same
  architecture for \(t=L\) internal updates. The stored reference results are
  exact for every start and every depth through 40.
- The grey dashed line is an explicit untied one-hop unrolling reference,
  \(2550L\) dense parameter slots. It is a reference construction, not a lower
  bound on feed-forward models.

## Main result

| Maximum supported depth | Median smallest MLP parameters | IQR |
|---:|---:|---:|
| 1 | 478 | 478-478 |
| 5 | 799 | 692-906 |
| 10 | 906 | 906-1,067 |
| 20 | 1,334 | 1,174-1,334 |
| 30 | 1,334 | 1,334-1,548 |
| 40 | 1,762 | 1,762-1,762 |

At depth 40, the learned shortcut MLP is 58 times smaller than the 102,000-slot
explicit unrolling. The naive linear unrolling curve is therefore not a valid
capacity claim once shortcuts are allowed.

## Reuse diagnostic

At depth 40, each selected MLP was retrained after withholding 10 of the 50
start nodes independently at every depth from 2 to 40. All depth-one queries
were retained, so the model still saw the complete pointer table.

- Directly supervised query accuracy: 99.6% (95% t interval: +/- 0.6 percentage
  points across 10 tables).
- Held-out start/depth accuracy: 62.8% (95% t interval: +/- 6.7 percentage
  points across 10 tables).
- Chance accuracy is 2%.
- The matched recurrent AC remains exact from one-hop training.

This separates two effects. A feed-forward MLP can compress many answers from a
seen table into a surprisingly small shortcut model, but that fit does not
provide the same reliable reuse as explicitly applying one learned transition
again. The defensible recurrence claim is therefore about reusable computation
and extension by additional updates, not an absolute parameter lower bound.

## Poster sentence

> Shortcuts compress seen 40-hop answers to 1,762 MLP parameters, 58 times below
> explicit unrolling, but held-out query accuracy falls to 62.8%; the fixed AC
> remains exact by reusing one learned transition for 40 updates.

The primary poster figure is `poster_ac_time_vs_mlp_size.*`. Each point directly
maps AC internal time on the x-axis to the smallest tested MLP size supporting
the same maximum pointer depth on the y-axis. The explicit-unrolling comparison
is retained separately as an analysis figure.

## Limitations

- This is an empirical frontier for the registered one-hidden-layer MLP family,
  not a lower bound for all feed-forward architectures.
- The main frontier measures fitted support on seen tables and uses direct
  supervision at every supported depth.
- MLP parameters, AC synapses and AC internal updates are not interchangeable
  physical units.
- The reduced held-out diagnostic uses one deterministic restart per table.

## Reproduction

Run from the assembly-calculus repository:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/thesis-poster-mpl \
  .venv/bin/python \
  scripts/thesis_c/resource/generate_learned_shortcut_frontier.py \
  --ac-root . \
  --output results/thesis_c/resource/learned_shortcuts \
  --workers 8 \
  --table-limit 10 \
  --holdout-restarts 1
```

The generator validates that its pointer tables exactly match
`results/thesis_c/resource/seen_time_size_raw.csv` before fitting or plotting.
