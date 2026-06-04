**Purpose:** classify which tasks should be sensitive to internal AC time, and explain what kind of time-sensitivity each class should show.

This note is a taxonomy. It connects [[Static Tasks]], [[Temporal and Iterative Tasks]], [[Error Accumulation]], and [[Time–Size Tradeoff]]. The aim is not to prove a universal complexity theorem for AC. The aim is to sort thesis experiments by the task-level dependency structure that internal time must handle.

## 1. Dependency Depth

For an input $q$, define the task-level dependency depth

$$
D(q)
$$

as the number of dependent computational transitions that must be executed after the relevant task information is available.

This is not the same as the AC update budget $t$. It is a property of the task representation. For static classification,

$$
D(q)=0.
$$

For pointer chasing with requested depth $L$,

$$
D(q)=L.
$$

If one symbolic transition takes approximately $c$ AC updates, then a temporal-reuse implementation needs

$$
t\gtrsim cD(q).
$$

This is only an execution-time condition. Reliability also requires

$$
D(q)\epsilon(c)\ll 1.
$$

Thus problem classes differ along two axes:

$$
\text{dependency depth}
$$

and

$$
\text{local transition reliability}.
$$

![problem class map](<Images/Problem Classes/problem_class_map.png>)

*Figure: schematic map of problem classes by dependency depth and expected time sensitivity.*

## 2. Static Classification

A static classification task has the form

$$
q\mapsto y.
$$

The label is determined by the cue. There is no required sequence of task transitions after the input is available:

$$
D(q)=0.
$$

Internal AC time can still matter because the representation evolves:

$$
X_q(t)=T^t(X_q(0);W^\star).
$$

For class assemblies $S_c$, the relevant quantity is the margin

$$
m_y(t)
=
o_y(t)-\max_{z\neq y}o_z(t).
$$

Success is predicted when

$$
m_y(t)>0.
$$

Expected behaviour:

$$
\text{accuracy rises while }m_y(t)\text{ improves, then saturates or drifts}.
$$

MNIST classification belongs here. It can be time-sensitive, but not because the label requires executing $L$ dependent transitions.

## 3. Pattern Completion and Denoising

Pattern completion is still usually a static target task:

$$
q_{\mathrm{corrupt}}\mapsto y
$$

or

$$
q_{\mathrm{partial}}\mapsto S_y.
$$

The difference from ordinary static classification is that the initial cap is incomplete or noisy. Let $o_y(0)$ be the initial overlap with the target assembly and let $o_y^\star$ be the stable recalled overlap. A settling model is

$$
o_y(t)
\approx
o_y^\star-\left(o_y^\star-o_y(0)\right)\rho^t,
\qquad
0<\rho<1.
$$

If the task requires overlap at least $b$, then the predicted completion time is

$$
t_{\mathrm{complete}}
\approx
\left\lceil
\frac{
\log\left(\frac{o_y^\star-b}{o_y^\star-o_y(0)}\right)
}{
\log \rho
}
\right\rceil,
$$

provided

$$
o_y(0)<b<o_y^\star.
$$

Here time is useful as settling time, not as algorithmic depth. Once completion has occurred, extra time is predicted to have little effect unless drift or interference begins.

## 4. Finite Temporal Memory

A finite temporal memory task depends on a bounded window of past inputs. For a sequence

$$
q=(x_1,\dots,x_T),
$$

the target may depend only on the last $h$ items:

$$
y=f(x_{T-h+1},\dots,x_T),
$$

where $h$ is fixed independently of $T$.

Then

$$
D(q)=O(h).
$$

If $h$ is small and fixed, the task can be handled by a fixed recurrent buffer or by spatial unrolling. The required time does not grow with total sequence length $T$ once the relevant window has been encoded.

Expected behaviour:

$$
t\text{ helps until the }h\text{-step memory has been encoded, then saturates}.
$$

This class is time-sensitive, but only with a bounded horizon.

## 5. Iterated Transition Problems

An iterated transition problem supplies a transition rule

$$
M:\mathcal S\to\mathcal S
$$

and asks for

$$
s_L=M^L(s_0).
$$

Here

$$
D(q)=L.
$$

With temporal reuse, the same transition mechanism is applied repeatedly:

$$
s_0\to s_1\to\cdots\to s_L.
$$

If one transition takes $c$ AC updates, then the execution condition is

$$
t\geq cL.
$$

If the per-transition error is $\epsilon(c)$, then path accuracy is approximately

$$
Acc_{\mathrm{path}}(L,c)
\approx
\left(1-\epsilon(c)\right)^L.
$$

Thus iterated transition tasks are strongly time-sensitive:

$$
\text{more internal time enables more dependent transitions}.
$$

Pointer chasing is the thesis benchmark for this class.

DFA evaluation also belongs here when the input length varies. If the automaton state is updated by

$$
s_{\ell+1}=\delta(s_\ell,x_{\ell+1}),
$$

then a length-$T$ input requires

$$
D(q)=T
$$

dependent state updates, unless the computation has been spatially unrolled or compressed into shortcuts.

## 6. Algorithmic and Graph Problems

Many algorithmic tasks reduce to repeated local updates. Examples include graph traversal, reachability by bounded walk length, planning, and repeated rule application.

Let $G=(V,E)$ be a graph and suppose the task requires following or discovering a path of length $L$. The computation can often be written as

$$
s_{\ell+1}=F_G(s_\ell),
$$

or more generally

$$
S_{\ell+1}=F_G(S_\ell),
$$

where $S_\ell$ may be a set of currently reachable states.

The relevant depth is the number of dependent update rounds:

$$
D(q)\approx L.
$$

If the algorithm can expand many states in parallel, one AC update block may represent a whole breadth-first layer. Then the depth is not the number of visited nodes, but the number of sequential expansion layers.

Expected behaviour:

$$
t\text{ should scale with algorithmic depth, not necessarily with input size alone}.
$$

This distinction matters. A large graph with a shallow required path can be easier than a smaller graph requiring many dependent transitions.

Binary search is a useful secondary example of this class. For a sorted array of length $N$, the symbolic state is the current interval $(a_\ell,b_\ell)$. Each step compares the query $x$ with the midpoint value and updates the interval. The worst-case dependency depth is

$$
D(q)
\leq
\lceil\log_2 N\rceil.
$$

Therefore the time prediction is not

$$
t\approx cN,
$$

but rather

$$
t\approx c\lceil\log_2 N\rceil,
$$

provided one comparison-and-interval-update step can be implemented in about $c$ AC updates. This makes binary search a clean comparison task for CLRS-style algorithmic models, while pointer chasing remains the cleaner primary test of arbitrary transition execution.

## 7. Lookup and Memorisation Tasks

A lookup task has a direct stored association:

$$
q\mapsto y.
$$

If the association is already learned or supplied as a direct table entry, then

$$
D(q)=0
$$

or at most a small constant. Internal time is then mainly recall and settling time.

Expected behaviour:

$$
\text{accuracy rises quickly with }t\text{ and then saturates}.
$$

This is different from unseen pointer chasing. In unseen pointer chasing, the model cannot solve the task by memorising

$$
(M,s_0,L)\mapsto s_L
$$

for every table. It must use the presented table representation and execute or simulate the transition process.

## 8. Shortcut-Encoded Iterative Tasks

Some tasks are iterative in principle but become shallow if the instance representation includes shortcuts. For pointer chasing, instead of representing only $M$, the instance may include

$$
M^2,\quad M^4,\quad M^8,\quad \dots.
$$

Then

$$
L=\sum_q b_q2^q
$$

can be executed using at most

$$
\lfloor\log_2 L\rfloor+1
$$

shortcut transitions.

This changes the effective dependency depth:

$$
D_{\mathrm{eff}}(q)
=
O(\log L)
$$

instead of

$$
D(q)=L.
$$

This is a time-size tradeoff. The price is larger instance representation size and shortcut-specific error:

$$
\epsilon_q(c,S)
=
\Pr[\widehat{M^{2^q}}_c(s)\neq M^{2^q}(s)].
$$

Shortcuts help only when the smaller number of steps outweighs the extra representation cost and any increase in shortcut error.

## 9. Non-Local Problems and Fixed-Time Limits

The thesis appendix argues that fixed-time explicit-input models can only depend on a bounded portion of the input. In this taxonomy, that corresponds to bounded dependency depth.

If a task family has unbounded dependency depth,

$$
D(q)\to\infty,
$$

then a fixed internal update budget cannot solve all depths by temporal reuse:

$$
t=O(1)
\quad\not\geq\quad
cD(q)
$$

for large $D(q)$.

This does not mean every large input is impossible in fixed time. It means the relevant dependency must either be bounded, precomputed, or represented as a shortcut. Otherwise the task needs internal time that grows with dependency depth.

## 10. Summary of Expected Time Effects

| Problem class | Dependency depth | Main role of $t$ | Predicted curve |
|---|---:|---|---|
| static classification | $0$ | margin settling or drift | rise, plateau, or fall |
| pattern completion / denoising | $0$ task depth, nonzero settling time | attractor completion | rise then plateau |
| finite temporal memory | $O(h)$ | encode bounded window | rise then plateau |
| lookup / memorisation | $0$ or $O(1)$ | recall stored association | quick rise then plateau |
| iterated transition | $L$ | execute dependent transitions | time-depth frontier |
| binary search | $O(\log N)$ | adaptive comparison and interval updates | threshold near $c\log_2 N$ |
| algorithmic / graph traversal | algorithmic depth | repeated local update rounds | accuracy tracks depth |
| shortcut-encoded iteration | $O(\log L)$ or $O(1)$ | execute stored shortcuts | less time, more size/error |

![time sensitivity curves](<Images/Problem Classes/time_sensitivity_curves.png>)

*Figure: schematic accuracy curves by problem class. Static and lookup tasks saturate; iterative tasks show execution thresholds; fixed-time runs fail as depth increases.*

## 11. Thesis Use

This taxonomy gives the experimental logic.

| Thesis question | Best task class |
|---|---|
| Does extra time help settling? | pattern completion or static classification |
| Does extra time cause drift? | confusing static classes |
| Does time execute computation? | iterated transition tasks |
| Do small errors compound? | long-depth pointer chasing |
| Can AC follow logarithmic algorithmic control? | binary search, secondary |
| Can size substitute for time? | shortcut-encoded transition tasks |
| Can fixed-time models handle non-local depth? | unbounded iterative or graph tasks |

The core claim is:

$$
\text{the effect of internal time depends on task dependency depth.}
$$

For static tasks, time changes a representation. For iterative tasks, time executes dependent transitions. For long iterative tasks, local transition errors accumulate with depth.

The schematic plots in this note are generated by [generate_problem_class_plots.py](<Plots/Problem Classes/generate_problem_class_plots.py>).
