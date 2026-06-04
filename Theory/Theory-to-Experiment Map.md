**Purpose:** connect each theory claim to concrete experiments, measurements, plots, and support criteria.

This note is the experimental contract for the theory stack. The experiments should not merely show that the models work. They should decide whether the proposed explanations are supported:

$$
\text{static time}
\rightarrow
\text{margin settling or drift}
$$

$$
\text{iterative time}
\rightarrow
\text{execution depth and error accumulation}
$$

$$
\text{size}
\rightarrow
\text{lower error, faster transitions, or shortcuts}.
$$

## 1. Experiment Set

The current three problem families are enough for the core thesis:

| Experiment | Problem class | Main reason to include |
|---|---|---|
| MNIST | static classification | tests margin settling, saturation, and drift |
| DFA | iterated state update over an input sequence | tests sequential state tracking with known transition rule |
| pointer chasing | iterated transition on unseen tables | tests internal execution time most directly |

Binary search is a useful secondary comparison benchmark, especially for comparison with CLRS-style algorithmic models, but it should not displace pointer chasing as the primary temporal experiment:

| Secondary experiment | Problem class | Main reason to include |
|---|---|---|
| binary search | logarithmic-depth adaptive search | tests algorithmic control with known $O(\log N)$ dependency depth |

No additional primary task family is needed now. Extra experiments should be ablations of the core three, or binary search as a secondary comparison, unless a specific theory claim remains untested.

Useful optional ablations:

| Optional ablation | Why it is useful |
|---|---|
| shortcut pointer chasing | directly tests the time-size tradeoff |
| larger pointer tables | tests whether transition error grows with state count |
| fixed-time baselines | tests the non-local / bounded-time argument |

## 2. Global Data Schema

Every run should save enough information to reconstruct accuracy, margins, and error trajectories. At minimum, record:

| Field | Meaning |
|---|---|
| `experiment` | `mnist`, `dfa`, `pointer_chasing`, or `binary_search` |
| `seed` | random seed |
| `theta_id` | identifier for fixed model instance $\theta_{\mathrm{inst}}$ |
| `n,k,p,beta` | AC hyperparameters |
| `t` | total internal update budget |
| `c` | updates per symbolic transition, where applicable |
| `L`, `T`, or `N` | pointer depth, DFA length, or binary-search array length |
| `instance_id` | MNIST image, DFA string, pointer table/start pair, or array/query pair |
| `target` | true label, accept state, final pointer state, or target index |
| `prediction` | model output |
| `correct` | Boolean correctness |
| `trajectory` | decoded state or class at each internal/macro step |
| `overlaps` | overlap vector with candidate assemblies |
| `margins` | relevant class or transition margins |

For temporal tasks, store the full decoded trajectory, not only the final answer. Without trajectories, error accumulation cannot be tested.

## 3. Common Protocol Rules

These rules should apply across MNIST, DFA, pointer chasing, and binary search.

| Rule | Reason |
|---|---|
| freeze evaluation weights at $W^\star$ | isolates internal time from continued learning |
| report mean and uncertainty over seeds | avoids single-run conclusions |
| keep training and evaluation instances separate | prevents lookup or memorisation explanations |
| store trajectories, not only final outputs | enables first-error and path-accuracy analysis |
| report both final accuracy and path accuracy for temporal tasks | separates final recovery from true trajectory correctness |
| define the readout time explicitly | avoids overshooting or hidden extra transitions |
| save raw overlaps or logits where possible | lets margin theory be checked after the run |

For each plotted curve, report either standard error or a bootstrap confidence interval. For a metric $Z$ over independent instances,

$$
\widehat{\mu}_Z
=
\frac{1}{m}\sum_{i=1}^{m}Z_i
$$

and

$$
\operatorname{SE}(\widehat{\mu}_Z)
=
\frac{\widehat{\sigma}_Z}{\sqrt m}.
$$

For seed-dependent experiments, aggregate by seed first, then report variation across seeds. Otherwise a large number of test examples can hide seed instability.

## 4. Claim-to-Experiment Matrix

| Theory claim | Primary experiment | Required measurement | Required plot |
|---|---|---|---|
| static time changes margins | MNIST | $o_y(t)$, $\max_{z\neq y}o_z(t)$, $m_y(t)$ | overlap and margin vs $t$ |
| static accuracy saturates | MNIST | global and per-class accuracy | $Acc_y(t)$ and $Acc(t)$ vs $t$ |
| too much time can cause drift | MNIST | wrong-class overlap and confusion pairs | confusion matrix vs $t$, pair plots |
| iterative tasks need execution time | pointer chasing | accuracy over $(L,t)$ | heatmap with boundary near $t=cL$ |
| one-step reliability improves with $c$ | pointer chasing, DFA | $\epsilon(c)$ | one-step error vs $c$ |
| long chains accumulate error | pointer chasing, DFA | path accuracy and first-error index | accuracy vs $L$, first-error histogram |
| fixed total time creates a depth cliff | pointer chasing | $Acc(L,t)$ for fixed $t$ | accuracy vs $L$ for several $t$ |
| time-size tradeoff | pointer chasing ablation | $Acc(S,t,L)$ or shortcut vs reuse | isoaccuracy curves over $(S,t)$ |
| DFA state tracking is sequential | DFA | state accuracy by input position | state accuracy vs position and length |
| logarithmic algorithmic depth | binary search | accuracy and comparison trace over $(N,t)$ | accuracy vs $\lceil\log_2 N\rceil$ and $t$ |

## 5. Fit-Then-Predict Protocol

The strongest test is to fit local quantities on small or one-step regimes, then predict held-out temporal depth. Do not fit the long-depth curves directly.

For pointer chasing, estimate an execution constant from the small-depth time threshold:

$$
t_{\mathrm{thr}}(L)
\approx
\widehat c_{\mathrm{exec}}L.
$$

Estimate one-step transition error separately:

$$
\widehat\epsilon(c)
=
\widehat{\Pr}
\left[
\widehat M_c(s)\neq M(s)
\right].
$$

Then predict held-out path accuracy at larger depths:

$$
\widehat{Acc}_{\mathrm{path}}(L,c)
=
\left(1-\widehat\epsilon(c)\right)^L.
$$

If transition difficulty changes with depth or state position, use the measured non-identical product:

$$
\widehat{Acc}_{\mathrm{path}}(L)
=
\prod_{\ell=0}^{L-1}
\left(1-\widehat\epsilon_\ell\right).
$$

For fixed total time, combine the executable-regime condition with the local error estimate:

$$
c(t,L)
=
\left\lfloor\frac{t}{L}\right\rfloor.
$$

$$
\widehat{Acc}_{\mathrm{path}}(L,t)
\approx
\mathbf 1\{c(t,L)\geq c_{\min}\}
\left(1-\widehat\epsilon(c(t,L))\right)^L.
$$

The fit set should use small depths, one-step transitions, and training seeds. The prediction set should use larger depths, unseen pointer tables or strings, and held-out seeds. The comparison plot should show predicted path accuracy against observed path accuracy with uncertainty intervals.

For binary search, use the same fit-then-predict logic with

$$
L_N=\lceil\log_2 N\rceil
$$

as the nominal comparison depth. Estimate per-comparison or per-interval-update error on small arrays, then predict accuracy on larger held-out arrays.

## 6. MNIST Experiment

MNIST is the static-task experiment. The target is

$$
q\mapsto y.
$$

The key margin is

$$
m_y(t)
=
o_y(t)-\max_{z\neq y}o_z(t).
$$

The theory predicts:

$$
\text{accuracy improves only while }m_y(t)\text{ improves}.
$$

It also predicts that visually similar classes may degrade if a competing overlap grows faster than the correct overlap.

### What To Sweep

| Variable | Suggested values | Purpose |
|---|---:|---|
| internal time $t$ | $0$ to $20$, or until curves settle | main independent variable |
| random seed | at least $5$ seeds | uncertainty estimate |
| recurrent on/off | if easy to implement | isolates recurrent drift |
| stimulus held vs cue-only | if supported | separates stimulus time from post-cue time |

Use the same trained weights when sweeping $t$:

$$
W(r)=W^\star.
$$

Evaluation plasticity should be frozen unless explicitly testing online adaptation.

### What To Measure

For each test image and each $t$:

| Measurement | Definition |
|---|---|
| correct overlap | $o_y(t)$ |
| strongest wrong overlap | $\max_{z\neq y}o_z(t)$ |
| margin | $m_y(t)$ |
| predicted class | $\hat y_t$ |
| correctness | $\mathbf 1\{\hat y_t=y\}$ |
| full class-overlap vector | $(o_0(t),\dots,o_9(t))$ |

Also compute per-class quantities:

$$
Acc_y(t)
=
\Pr[\hat y_t=y\mid y]
$$

and margin quantiles:

$$
Q_{0.1}[m_y(t)\mid y].
$$

### What To Plot

Minimum MNIST plots:

| Plot | Why it matters |
|---|---|
| global $Acc(t)$ vs $t$ | shows saturate or degrade pattern |
| per-class $Acc_y(t)$ vs $t$ | identifies fragile classes |
| $o_y(t)$ and $\max_{z\neq y}o_z(t)$ vs $t$ | tests margin theory |
| $m_y(t)$ mean and lower quantiles vs $t$ | checks tail failures |
| confusion matrix at early, best, and late $t$ | shows drift pairs |
| pair plots for $7/9$, $3/5$, $4/9$ if present | explains class-specific degradation |
| digit templates or active pixels vs $t$ | qualitative support only |

Support criterion:

$$
Acc_y(t)\text{ should track the distribution of }m_y(t).
$$

If accuracy changes without corresponding margin changes, the readout or overlap model is missing something.

## 7. DFA Experiment

DFA evaluation is a controlled sequential state-tracking task. For input symbols $x_1,\dots,x_T$,

$$
s_{\ell+1}=\delta(s_\ell,x_{\ell+1}).
$$

The dependency depth is

$$
D(q)=T.
$$

The DFA experiment is useful because the correct intermediate state is known at every step. This makes it easier than pointer chasing to diagnose where errors begin.

### What To Sweep

| Variable | Suggested values | Purpose |
|---|---:|---|
| sequence length $T$ | $0$ to maximum feasible length | tests depth generalisation |
| presentations / training examples | existing sweep plus fixed checkpoints | separates learning from execution |
| updates per symbol $c$ | $1,2,4,8$ if supported | tests one-step reliability |
| random seed | at least $5$ seeds | uncertainty estimate |

If the implementation presents one symbol at a time, report both total internal updates and per-symbol updates:

$$
t_{\mathrm{total}}\approx cT.
$$

### What To Measure

For each string:

| Measurement | Definition |
|---|---|
| final accept/reject accuracy | correct final DFA decision |
| state accuracy at position $\ell$ | $\Pr[\hat s_\ell=s_\ell]$ |
| path accuracy | all intermediate states correct |
| first-error index | first $\ell$ with $\hat s_\ell\neq s_\ell$ |
| transition confusion | predicted next state for each $(s,x)$ |

Define one-step transition error:

$$
\epsilon_{s,x}(c)
=
\Pr[\widehat{\delta}_c(s,x)\neq \delta(s,x)].
$$

Average over visited transitions:

$$
\epsilon(c)
=
\mathbb E_{(s,x)}[\epsilon_{s,x}(c)].
$$

### What To Plot

Minimum DFA plots:

| Plot | Why it matters |
|---|---|
| final accuracy vs sequence length $T$ | tests length generalisation |
| state accuracy vs position $\ell$ | shows sequential degradation |
| path accuracy vs $T$ | tests error accumulation |
| first-error histogram | tests constant-error vs worsening-error model |
| transition confusion matrix for $(s,x)\mapsto s'$ | diagnoses local transition failures |
| heatmap: training presentations vs test length | separates learning sufficiency from depth |
| $\epsilon(c)$ vs $c$ | tests whether more per-symbol time improves reliability |

Support criterion:

$$
Acc_{\mathrm{path}}(T)
\approx
\left(1-\epsilon(c)\right)^T
$$

when per-step error is stable. If performance stays perfect for all tested $T$, then the DFA experiment mainly supports learnability, not the time-depth frontier. In that case, stress the DFA by increasing $T$, lowering $c$, reducing training presentations, or increasing automaton complexity. If it remains perfect, pointer chasing remains the stronger temporal benchmark.

## 8. Binary Search Experiment

Binary search is a secondary temporal benchmark. It is useful because the target algorithm is standard, the dependency depth is logarithmic in input size, and results can be compared against CLRS-style algorithmic models.

For a sorted array $A$ of length $N$ and query $x$, the task is

$$
q=(A,x,N),
$$

with target index

$$
i^\star
=
\operatorname{index}(A,x)
$$

or a not-found symbol. The symbolic state is the current search interval $(a_\ell,b_\ell)$. One transition computes

$$
m_\ell
=
\left\lfloor\frac{a_\ell+b_\ell}{2}\right\rfloor
$$

and updates

$$
(a_{\ell+1},b_{\ell+1})
=
\begin{cases}
(a_\ell,m_\ell-1), & x<A[m_\ell],\\
(m_\ell+1,b_\ell), & x>A[m_\ell],\\
(m_\ell,m_\ell), & x=A[m_\ell].
\end{cases}
$$

The nominal dependency depth is

$$
D(q)
\leq
L_N
=
\lceil\log_2 N\rceil.
$$

Thus a temporal-reuse AC implementation should show an execution threshold near

$$
t
\gtrsim
cL_N,
$$

not near $cN$. This makes binary search a clean secondary test of algorithmic depth rather than raw input size.

### What To Sweep

| Variable | Suggested values | Purpose |
|---|---:|---|
| array length $N$ | powers of two if possible | tests $O(\log N)$ depth scaling |
| internal time $t$ | below and above $c\lceil\log_2 N\rceil$ | tests execution threshold |
| updates per comparison $c$ | $1,2,4,8$ if supported | tests local comparison reliability |
| query type | present and absent keys | separates found/not-found behaviour |
| held-out arrays | unseen sorted arrays | avoids memorised lookup |

### What To Measure

| Measurement | Definition |
|---|---|
| final index accuracy | $\Pr[\hat i=i^\star]$ |
| comparison accuracy | correct branch at each midpoint |
| interval accuracy | $\Pr[(\hat a_\ell,\hat b_\ell)=(a_\ell,b_\ell)]$ |
| path accuracy | all interval states correct |
| first-error index | first wrong comparison or interval update |
| readout type | exact index, nearest index, or not-found |

Define local binary-search error as

$$
\epsilon_{\mathrm{bs}}(c)
=
\Pr[
\widehat{(a_{\ell+1},b_{\ell+1})}
\neq
(a_{\ell+1},b_{\ell+1})
].
$$

Then the predicted path accuracy is

$$
Acc_{\mathrm{path}}(N,c)
\approx
\left(1-\epsilon_{\mathrm{bs}}(c)\right)^{\lceil\log_2 N\rceil}.
$$

### What To Plot

| Plot | Why it matters |
|---|---|
| accuracy vs $t$ for fixed $N$ | shows execution threshold |
| accuracy vs $N$ for fixed $t$ | tests logarithmic depth cliff |
| heatmap $Acc(N,t)$ | should align with $t\approx c\lceil\log_2 N\rceil$ |
| path accuracy vs $\lceil\log_2 N\rceil$ | tests error accumulation |
| first-error histogram | diagnoses whether errors are early or uniform |
| comparison/branch confusion | diagnoses local comparison failures |

Support criterion:

$$
N_{\max}(t)
\approx
2^{\lfloor t/c\rfloor}
$$

until local comparison error dominates. If accuracy scales with $N$ rather than $\log_2 N$, the implementation may not be executing binary search as a temporal algorithm.

## 9. Pointer Chasing Experiment

Pointer chasing is the main experiment for internal execution time. Each instance supplies

$$
q=(M,s_0,L),
$$

with target

$$
s_L=M^L(s_0).
$$

The key theory prediction is:

$$
t\geq cL
$$

and

$$
L\epsilon(c)\ll 1.
$$

### What To Sweep

| Variable | Suggested values | Purpose |
|---|---:|---|
| depth $L$ | broad range, including failure region | tests time-depth frontier |
| internal time $t$ | below and above $cL$ | tests execution threshold |
| updates per transition $c$ | $1,2,4,8$ or inferred from $t/L$ | tests reliability tradeoff |
| table size $N$ | at least two values if feasible | tests state-space scaling |
| model size $S$ | vary $n$, $k$, or area count if feasible | tests time-size tradeoff |
| random seed / unseen table | many independent tables | avoids memorisation |

Use unseen tables for evaluation. Otherwise the experiment can become lookup or memorisation.

### What To Measure

For each table, start state, depth, and time budget:

| Measurement | Definition |
|---|---|
| final-state accuracy | $\Pr[\hat s_L=s_L]$ |
| path accuracy | all decoded intermediate states match true path |
| one-step error | $\epsilon(c)$ |
| first-error index | first wrong transition |
| decoded state sequence | $\hat s_0,\dots,\hat s_L$ |
| true state sequence | $s_0,\dots,s_L$ |
| state overlap | $o_s(r)$ for candidate states |
| transition margin | $D_{s,v}^{(c)}$ |
| transition confusion | $P_c^M(v\mid s)$ |

Separate final-state accuracy from path accuracy:

$$
\Pr[\text{no transition error}]
\leq
\Pr[\hat s_L=s_L].
$$

For path-novel instances, the two should be close. If they are not close, collisions or recovery are important and must be reported.

### What To Plot

Minimum pointer-chasing plots:

| Plot | Why it matters |
|---|---|
| heatmap $Acc(L,t)$ | should show boundary near $t=cL$ |
| final accuracy vs $t$ for fixed $L$ | shows execution threshold |
| final accuracy vs $L$ for fixed $t$ | shows depth cliff |
| path accuracy vs $L$ for fixed $c$ | tests $(1-\epsilon)^L$ |
| one-step error $\epsilon(c)$ vs $c$ | tests transition settling |
| first-error index histogram | tests error accumulation |
| path accuracy vs final accuracy | estimates recovery/collisions |
| transition confusion matrix | identifies crosstalk states |
| margin before first error | tests transition-margin theory |

Support criterion:

$$
L_{\max}(t)
\approx
\left\lfloor\frac{t}{c}\right\rfloor
$$

until error accumulation dominates.

If accuracy improves with $t$ but does not shift with $L$ according to a time-depth boundary, then the model may be settling a representation rather than executing repeated transitions.

## 10. Time-Size Tradeoff Experiment

This is optional but important if the thesis explicitly claims a time-size tradeoff. It can be implemented as a pointer-chasing ablation rather than a new task.

There are two clean versions.

### Version A: Vary Model Size

Hold the task format fixed and vary model size:

$$
S_{\mathrm{model}}\in\{S_1,S_2,\dots\}.
$$

Use size measures from [[AC Mathematical Setup]]:

$$
S_{\mathrm{neurons}}=an
$$

and

$$
S_{\mathrm{synapses}}\approx |\mathcal F|pn^2.
$$

Measure:

$$
Acc_{\mathcal Q_L}(S,t).
$$

Plot isoaccuracy contours over

$$
(S,t).
$$

Support criterion: larger $S$ should reduce required $t$ only if it lowers transition error, lowers per-transition time $c$, or reduces crosstalk.

### Version B: Shortcut Pointer Chasing

Compare temporal reuse with shortcut-encoded instances.

Temporal reuse supplies only

$$
M.
$$

Shortcut instances supply

$$
M^2,\quad M^4,\quad M^8,\quad \dots.
$$

Measure shortcut-specific error:

$$
\epsilon_q(c,S)
=
\Pr[\widehat{M^{2^q}}_c(s)\neq M^{2^q}(s)].
$$

Plot:

| Plot | Purpose |
|---|---|
| temporal reuse vs shortcut accuracy over $L$ | tests whether shortcuts reduce effective depth |
| runtime or update budget needed for target accuracy | measures time saving |
| instance size vs accuracy | shows the memory price |
| $\epsilon_q$ vs $q$ | checks whether large shortcuts are reliable |

This is the cleanest direct test of [[Time–Size Tradeoff]], but it is not required before the core pointer-chasing result is established.

## 11. Are More Experiments Needed?

No additional core experiment family is needed now. Binary search is justified as a secondary comparison benchmark, not as a replacement for the pointer-chasing core.

The three current families cover the theory:

| Theory component | Covered by |
|---|---|
| static margin and drift | MNIST |
| sequential state update | DFA |
| internal execution depth | pointer chasing |
| error accumulation | pointer chasing and DFA |
| logarithmic algorithmic control | binary search, secondary |
| time-size tradeoff | pointer-chasing size or shortcut ablation |

Do not add graph traversal or planning yet unless pointer chasing is already complete and the thesis needs an external demonstration. Graph tasks are conceptually useful, but they add implementation complexity without testing a fundamentally different theory claim.

The recommended additions are controlled ablations plus the binary-search comparison if implementation time permits:

- shortcut pointer chasing if the thesis needs a direct time-size tradeoff result;
- binary search if the thesis needs comparison with CLRS-style algorithmic models;
- model-size sweeps if compute budget allows.

## 12. Minimum Figure Package

The thesis should contain the following minimum figures.

### MNIST

| Figure | Required content |
|---|---|
| MNIST accuracy vs $t$ | global and per-class |
| MNIST margin vs $t$ | $o_y(t)$, strongest wrong overlap, $m_y(t)$ |
| MNIST confusion vs $t$ | early, best, late time |
| MNIST pair drift | confusing pairs such as $7/9$ or $3/5$ |

### DFA

| Figure | Required content |
|---|---|
| DFA accuracy vs sequence length | final accept/reject and state accuracy |
| DFA heatmap | training presentations vs test length |
| DFA first-error histogram | if trajectory data is available |
| DFA transition confusion | local transition failure analysis |

### Binary Search

| Figure | Required content |
|---|---|
| binary-search heatmap $Acc(N,t)$ | optional; boundary should track $t\approx c\lceil\log_2 N\rceil$ |
| binary-search path accuracy vs $\lceil\log_2 N\rceil$ | optional; tests accumulated comparison/update errors |
| binary-search first-error histogram | optional; diagnoses branch or interval-update failures |

### Pointer Chasing

| Figure | Required content |
|---|---|
| pointer heatmap $Acc(L,t)$ | diagonal time-depth boundary |
| pointer accuracy vs $t$ at fixed $L$ | execution threshold |
| pointer accuracy vs $L$ at fixed $t$ | depth cliff |
| path accuracy vs $(1-\epsilon)^L$ | error accumulation comparison |
| first-error histogram | localises failure point |
| transition margin before error | links AC margin to failure |

### Time-Size

| Figure | Required content |
|---|---|
| isoaccuracy over $(S,t)$ | only if model-size sweep is run |
| shortcut vs reuse | only if shortcut ablation is run |
| $\epsilon_q$ vs shortcut power | only if shortcuts are supplied |

## 13. Support and Falsification Criteria

The results support the theory if:

- MNIST changes with $t$ are explained by overlap margins and confusion pairs;
- temporal experiments show the expected dependence on dependency depth;
- pointer chasing shows a boundary near $t=cL$;
- binary search, if run, shows scaling closer to $\log_2 N$ than $N$;
- one-step error predicts long-depth path accuracy through $(1-\epsilon)^L$ or the measured non-identical product;
- larger models help only through lower $\epsilon$, smaller $c$, or shortcut structure.

The results challenge the theory if:

- static-task accuracy changes without measurable margin change;
- pointer chasing solves large $L$ with fixed small $t$ and no shortcuts;
- binary search succeeds at large $N$ without enough comparison steps or an identifiable shortcut;
- long-depth accuracy remains high despite measured nonzero one-step error;
- model size improves temporal depth without changing error, transition time, or representation structure;
- final-state accuracy greatly exceeds path accuracy without collisions or recovery explaining it.

In that case, the next step is not to discard the theory immediately. The first step is to inspect the trajectory data, because most disagreements should appear as a missing measurement: unobserved shortcut use, recovery, readout effects, or an incorrect estimate of $c$.

## 14. Final Experimental Priority

Priority order:

1. Pointer chasing heatmap over $(L,t)$.
2. Pointer chasing path accuracy, first-error index, and $\epsilon(c)$.
3. MNIST margin and confusion analysis over $t$.
4. DFA length and state-trajectory analysis.
5. Optional pointer-chasing size or shortcut ablation.
6. Optional binary search for CLRS-style comparison.

The thesis can stand without new task families if these are measured cleanly. The main requirement is not more experiments; it is recording the right internal variables so the plots can test the theory directly.
