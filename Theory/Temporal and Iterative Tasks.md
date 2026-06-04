**Purpose:** derive why temporal and iterative tasks need internal execution time, and give testable AC predictions for accuracy as a function of update budget and transition depth.

This note treats pointer chasing as the main example. The derivations are mean-field AC theory: they start from the $k$-cap update, then pass through transition margins, one-step error, and multi-step error accumulation. They are not exact theorems for every random graph realisation.

## 1. Iterative Task Setup

Let

$$
\mathcal S=\{1,\dots,N\}
$$

be a finite state set. A temporal transition task supplies a transition rule

$$
M:\mathcal S\to\mathcal S,
$$

a start state $s_0$, and a depth $L$. The state sequence is

$$
s_{\ell+1}=M(s_\ell),
\qquad
\ell=0,\dots,L-1.
$$

The target output is

$$
s_L=M^L(s_0).
$$

For pointer chasing, the task input is

$$
q=(M,s_0,L),
$$

and the label is the final state $s_L$. The depth-indexed task distribution is written $\mathcal Q_L$, and performance is

$$
Acc_{\mathcal Q_L}(\theta_{\mathrm{inst}},t)
=
\Pr_{(M,s_0,s_L)\sim\mathcal Q_L}
\left[
\hat s_t=s_L
\right].
$$

The key distinction is

$$
\text{task depth }L
\neq
\text{internal update budget }t.
$$

Depth $L$ is specified by the problem. Budget $t$ is the number of AC updates available to execute the computation.

## 2. Why Iterative Tasks Are Not Static Tasks

A static task has a target of the form

$$
q\mapsto y.
$$

After the cue is supplied, extra internal time can only transform the representation of the same input. By contrast, an iterative task requires a chain of dependent states:

$$
s_0\to s_1\to\cdots\to s_L.
$$

The dependence is sequential. The model cannot execute transition $\ell+1$ until it has formed the state needed by transition $\ell$:

$$
s_{\ell+2}=M(s_{\ell+1}).
$$

Thus the reusable computation has the form

$$
s_L
=
\underbrace{M(M(\cdots M}_{L\text{ times}}(s_0)\cdots)).
$$

If a temporal-reuse AC mechanism can execute only $R$ transition applications, then the natural reached state is

$$
M^R(s_0),
$$

not $M^L(s_0)$.

This does not rule out spatially unrolled circuits for bounded depths. The claim is narrower: if the same AC transition mechanism is reused over internal time, then dependent transition depth consumes internal update budget.

This note therefore extends [[Static Tasks]] rather than replacing it. The local AC mechanism is the same, but the measured object changes.

| Quantity | Static task | Temporal or iterative task |
|---|---|---|
| AC update | one representation trajectory $X_q(t)=T^t(X_q(0))$ | repeated transition trajectory $s_{\ell+1}=M(s_\ell)$ |
| overlap | class overlap $o_c(t)$ | state overlap $o_s(r)$ |
| margin | readout margin $m_y(t)$ | transition margin $D_{s,v}^{(c)}$ and state margin $m_\ell(r)$ |
| effect of extra time | improves or damages one readout margin | increases executable depth; per-transition budget can improve transition reliability |
| predicted curve | rise, plateau, or drift-induced fall | time-depth frontier plus error accumulation |

## 3. AC State Representation

Assume each symbolic state $s\in\mathcal S$ is represented by a state assembly

$$
S_s\subseteq\{1,\dots,n\},
\qquad
|S_s|=k
$$

in a current-state area $A$.

At internal update $r$, let

$$
K(r)=\operatorname{supp}(x_A(r))
$$

be the active cap. The overlap with state assembly $S_s$ is

$$
o_s(r)
=
\frac{|K(r)\cap S_s|}{k}.
$$

The decoded state is

$$
\hat s(r)
=
\arg\max_{s\in\mathcal S}o_s(r).
$$

For the true state $s_\ell$, define the state margin

$$
m_\ell(r)
=
o_{s_\ell}(r)
-
\max_{v\neq s_\ell}o_v(r).
$$

A clean no-tie representation of $s_\ell$ satisfies

$$
m_\ell(r)>0.
$$

If $m_\ell(r)=0$, correctness depends on the tie-breaking rule.

## 4. One AC Transition From the k-Cap Rule

Suppose the current state is $s$ and the correct next state is

$$
v=M(s).
$$

The transition module should move activity from $S_s$ toward $S_v$.

Let $W^{\mathrm{eff}}(M,r)$ denote the effective transition operator when the pointer table $M$ is present at internal update $r$. The trained weights $W^\star$ are fixed after training. The notation $W^{\mathrm{eff}}(M,r)$ does not mean the model receives a newly trained weight matrix for each table. It means the operator induced by the fixed weights, the presented table representation, and the gating or control schedule at that update.

Ignoring area labels, the synaptic input to neuron $i$ during one internal transition update is

$$
I_i(r)
=
\sum_{j\in K(r)}
W_{ji}^{\mathrm{eff}}(M,r)
+u_i^M(r),
$$

where $u_i^M(r)$ denotes any table-dependent external drive or gating signal.

Let $\tau_k(r)$ be the $k$-cap threshold:

$$
\tau_k(r)
=
\operatorname{kthlargest}\{I_1(r),\dots,I_n(r)\}.
$$

The next cap is

$$
K(r+1)
=
\{i:I_i(r)\geq \tau_k(r)\},
$$

up to ties. Therefore the next overlap with candidate state assembly $S_v$ is

$$
o_v(r+1)
=
\frac{1}{k}
\sum_{i\in S_v}
\mathbf 1\{I_i(r)\geq \tau_k(r)\}.
$$

This is the exact AC bridge: a state becomes represented when its assembly neurons receive enough input to enter the next $k$-cap.

## 5. Population Threshold and Transition Signal Margin

The threshold $\tau_k(r)$ is not a free parameter. It is determined by the full population. Since exactly $k$ of $n$ neurons enter the cap,

$$
\frac{k}{n}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1\{I_i(r)\geq \tau_k(r)\},
$$

apart from tie handling.

In mean field, divide neurons into groups $g\in\mathcal G$, such as the correct destination assembly, wrong destination assemblies, and background neurons. Let $\pi_g$ be the population fraction of group $g$. Then $\tau_k(r)$ approximately solves

$$
\frac{k}{n}
\approx
\sum_{g\in\mathcal G}
\pi_g
\Pr[I^{(g)}(r)\geq \tau_k(r)].
$$

Under a normal input approximation,

$$
\frac{k}{n}
\approx
\sum_{g\in\mathcal G}
\pi_g
\Phi\left(
\frac{\mu_g(r)-\tau_k(r)}
{\sigma_g(r)}
\right).
$$

For candidate next state $v$,

$$
\mathbb E[o_v(r+1)]
\approx
\Pr[I^{(v)}(r)\geq \tau_k(r)].
$$

Thus

$$
\mathbb E[o_v(r+1)]
\approx
\Phi\left(
\frac{\mu_v(s,r)-\tau_k(r)}
{\sigma_v(s,r)}
\right).
$$

Define the transition signal margin

$$
\gamma_{s\to v}(r)
=
\mu_v(s,r)-\tau_k(r).
$$

Increasing $\gamma_{s\to M(s)}(r)$ increases the expected overlap with the correct next-state assembly. Increasing $\gamma_{s\to v}(r)$ for $v\neq M(s)$ increases wrong-destination overlap.

![transition signal margin](<Images/Temporal and Iterative Tasks/transition_signal_margin.png>)

*Figure: schematic relation between transition signal margin and expected next-state overlap. A larger positive margin makes the destination assembly more likely to enter the next $k$-cap.*

## 6. Synaptic Transition Crosstalk

Define the average directed transition strength from current-state assembly $S_a$ into destination assembly $S_v$:

$$
B_{a\to v}^{M,r}
=
\frac{1}{k^2}
\sum_{j\in S_a}
\sum_{i\in S_v}
W_{ji}^{\mathrm{eff}}(M,r).
$$

Absent edges contribute zero. Therefore $B_{a\to v}^{M,r}$ already includes the sparsity of the AC graph. It is an effective all-pairs synaptic support measure, not an average conditional on an edge existing.

For a clean transition from $s$ to $M(s)$, the desired inequality is

$$
B_{s\to M(s)}^{M,r}
>
B_{s\to v}^{M,r}
\qquad
\text{for }v\neq M(s).
$$

If the current cap has overlaps $o_a(r)$ with state assemblies $S_a$, then a first-order mean input model is

$$
\mu_v(s,r)
\approx
k\left[
w_0+
\sum_{a\in\mathcal S}
o_a(r)
\left(B_{a\to v}^{M,r}-w_0\right)
\right]
+u_v^M(r).
$$

Here $w_0$ is a background all-pairs average, also including absent edges as zeros. There is no extra multiplicative $p$ in this convention, because graph sparsity has already been absorbed into $B_{a\to v}^{M,r}$ and $w_0$.

This assumes approximately exchangeable neurons within each state assembly and treats $B_{a\to v}^{M,r}$ as an effective average weight. It exposes two error mechanisms:

- **destination crosstalk:** $B_{s\to v}^{M,r}$ is large for a wrong destination $v$;
- **state crosstalk:** the current cap has nonzero overlap with a wrong state $a$, and $B_{a\to v}^{M,r}$ supports the wrong destination.

Structural assembly overlap,

$$
\Omega_{a v}
=
\frac{|S_a\cap S_v|}{k},
$$

is only one source of interference. The directed quantity $B_{a\to v}^{M,r}$ is usually more important for transition errors because it measures actual synaptic support.

## 7. From One Update to One Symbolic Transition

One symbolic transition may require more than one AC update. Let $c$ be the number of internal updates allocated to one transition. Define the macro-step index

$$
r_\ell=\ell c.
$$

Let $F_{M,h}$ be the mean-field one-update overlap map at substep $h$ of the transition schedule:

$$
o(r+h+1)
=
F_{M,h}(o(r+h)),
\qquad
h=0,\dots,c-1.
$$

If the same $c$-substep schedule is reused for each symbolic transition, define the macro-transition map

$$
\mathcal T_M^{(c)}
=
F_{M,c-1}\circ\cdots\circ F_{M,0}.
$$

Then after $c$ internal updates,

$$
o(r_\ell+c)
=
\mathcal T_M^{(c)}(o(r_\ell)).
$$

If the one-update map is stationary across substeps, this reduces to $F_M^c$. The composition form is more precise when gating, table access, or effective weights depend on the substep.

For an ideal transition executor,

$$
\mathcal T_M^{(c)}(e_s)
\approx
e_{M(s)},
$$

where $e_s$ denotes a clean overlap vector concentrated on state $s$.

For a candidate wrong destination $v\neq M(s)$, define the pairwise transition margin after $c$ updates:

$$
D_{s,v}^{(c)}
=
o_{M(s)}(r+c)-o_v(r+c).
$$

The one-step transition is decoded correctly when

$$
\min_{v\neq M(s)}
D_{s,v}^{(c)}
>0.
$$

Therefore the one-step transition error is

$$
\epsilon_s(c)
=
\Pr\left[
\min_{v\neq M(s)}
D_{s,v}^{(c)}
\leq 0
\right].
$$

If $D_{s,v}^{(c)}$ has mean $\delta_{s,v}(c)$ and standard deviation $\eta_{s,v}(c)$, then a normal pairwise approximation gives

$$
\Pr[D_{s,v}^{(c)}\leq 0]
\approx
\Phi\left(
-\frac{\delta_{s,v}(c)}
{\eta_{s,v}(c)}
\right).
$$

By a union bound,

$$
\epsilon_s(c)
\leq
\sum_{v\neq M(s)}
\Phi\left(
-\frac{\delta_{s,v}(c)}
{\eta_{s,v}(c)}
\right).
$$

This connects AC overlap margins to transition accuracy. Larger destination margins reduce one-step error; more competing states increase the union-bound pressure.

## 8. Settling Model for One-Step Error

Near a stable transition target, the mean-field map can be linearised. For a relevant margin coordinate $d(c)$,

$$
d(c+1)-d^\star
\approx
\rho\left(d(c)-d^\star\right),
\qquad
0<\rho<1.
$$

Thus

$$
d(c)
\approx
d^\star-\left(d^\star-d(0)\right)\rho^c.
$$

As the margin rises, one-step error falls. A compact fitted form is

$$
\epsilon(c)
\approx
\epsilon_\infty+\epsilon_0\rho_\epsilon^c,
\qquad
0<\rho_\epsilon<1.
$$

Here $\epsilon_\infty$ is the irreducible transition error from finite capacity, crosstalk, noise, or imperfect table encoding. The decaying term is the part of the error reduced by giving the transition more internal settling time.

![per-transition budget](<Images/Temporal and Iterative Tasks/per_transition_budget.png>)

*Figure: schematic transition margin and one-step error as the per-transition AC budget $c$ increases. More updates help only until the irreducible error floor dominates.*

## 9. Time Budget and Executable Depth

If one symbolic transition consumes $c$ internal updates, then total budget $t$ permits approximately

$$
R(t)
=
\left\lfloor\frac{t}{c}\right\rfloor
$$

transition applications.

A depth-$L$ sequential executor needs

$$
R(t)\geq L,
$$

equivalently

$$
t\geq cL.
$$

This is the first temporal-task constraint.

When $t\geq cL$, assume evaluation reads out after the requested $L$ symbolic transitions. Extra unused updates are either not run, or are gated so that they do not apply additional pointer transitions. Without a stopping or readout control based on $L$, running for $R(t)>L$ would overshoot the requested answer and produce

$$
M^{R(t)}(s_0)
$$

instead of $M^L(s_0)$.

If $R(t)<L$ and the model simply reports the state it has reached, then success requires

$$
M^{R(t)}(s_0)=M^L(s_0).
$$

For path-novel pointer-chasing instances with no cycle or collision before depth $L$, this cannot happen when $R(t)<L$. In random finite tables it can happen by collision, but it is a coincidence rather than successful execution of the requested depth.

![time-depth boundary](<Images/Temporal and Iterative Tasks/time_depth_boundary.png>)

*Figure: predicted accuracy over depth $L$ and update budget $t$. The diagonal boundary reflects the execution-time condition $t\geq cL$; the fading above the boundary reflects accumulated transition errors.*

## 10. Multi-Step Error Accumulation

Assume the model has enough time to execute all $L$ transitions. Let $E_\ell$ be the event that transition $\ell$ is decoded correctly, conditioned on all previous decoded states being correct.

The no-error trajectory event is

$$
E_0\cap E_1\cap\cdots\cap E_{L-1}.
$$

Therefore

$$
\Pr[\text{no transition error}]
=
\prod_{\ell=0}^{L-1}
\Pr[E_\ell\mid E_0,\dots,E_{\ell-1}].
$$

If each conditional transition error is at most $\epsilon$, then

$$
\Pr[\text{no transition error}]
\geq
(1-\epsilon)^L.
$$

Since a no-error trajectory implies the correct final state,

$$
Acc_{\mathcal Q_L}(\theta_{\mathrm{inst}},t)
\geq
(1-\epsilon)^L
$$

within the enough-time regime, ignoring failures outside the model assumptions. This is a conservative path-correctness lower bound, not an exact equality.

The inequality can be strict because a trajectory with an intermediate error may still end at the correct final state:

$$
\Pr[\text{no transition error}]
\leq
\Pr[\hat s_L=s_L].
$$

For path-novel pointer-chasing instances, such recovery should be rare because a wrong intermediate state usually sends the computation along a different pointer path.

The complementary union bound is

$$
\Pr[\text{at least one transition error}]
\leq
L\epsilon.
$$

For small $\epsilon$,

$$
(1-\epsilon)^L
\approx
e^{-\epsilon L}.
$$

The second temporal-task constraint is therefore

$$
L\epsilon(c)\ll 1.
$$

Even if $t\geq cL$, depth can fail because small one-step errors compound.

![error accumulation](<Images/Temporal and Iterative Tasks/error_accumulation.png>)

*Figure: no-error trajectory probability falls with depth. The approximation $e^{-\epsilon L}$ explains why small one-step errors become visible in long pointer chains.*

## 11. Confusion-Matrix View

The product bound counts only perfectly correct trajectories. A more detailed model uses the table-conditioned transition confusion matrix

$$
P_c^M(v\mid s)
=
\Pr[\widehat M_c(s)=v],
$$

where $\widehat M_c(s)$ is the decoded state after $c$ updates from current state $s$.

For a fixed table $M$, the decoded-state distribution evolves as

$$
\hat\pi_{\ell+1}(v)
=
\sum_{s\in\mathcal S}
\hat\pi_\ell(s)P_c^M(v\mid s),
$$

with

$$
\hat\pi_0(s)=\mathbf 1\{s=s_0\}.
$$

The exact final-state success probability under this Markov approximation is

$$
\Pr[\hat s_L=s_L]
=
\hat\pi_L(s_L).
$$

This model allows recovery by collisions or cycles. The no-error product bound is stricter because it requires every transition along the true path to be correct.

## 12. Accuracy as a Function of Time and Depth

Combining execution time and error accumulation gives the central prediction. With fixed per-transition budget $c$,

$$
Acc_{\mathcal Q_L}(\theta_{\mathrm{inst}},t)
\approx
\mathbf 1\{R(t)<L\}A_{\text{early}}(L,t)
+
\mathbf 1\{R(t)\geq L\}
\left(1-\epsilon(c)\right)^L.
$$

Here $A_{\text{early}}(L,t)$ accounts for chance, collisions, or task shortcuts when $R(t)<L$. In path-novel pointer-chasing experiments, this term should be near baseline.

If the total budget $t$ is distributed across the required $L$ transitions, then

$$
c(t,L)
=
\left\lfloor\frac{t}{L}\right\rfloor,
$$

Let $c_{\min}$ be the minimum update budget needed to attempt one symbolic transition under the chosen protocol. If $c(t,L)<c_{\min}$, the requested computation is not executable by this transition schedule. A simple prediction is

$$
Acc_{\mathcal Q_L}(\theta_{\mathrm{inst}},t)
\approx
\mathbf 1\{c(t,L)<c_{\min}\}A_{\text{early}}(L,t)
+
\mathbf 1\{c(t,L)\geq c_{\min}\}
\left(1-\epsilon(c(t,L))\right)^L.
$$

For target accuracy $\alpha$, the usable depth is limited by both constraints:

$$
L
\leq
\left\lfloor\frac{t}{c}\right\rfloor
$$

and

$$
L
\lesssim
\frac{-\log \alpha}{\epsilon(c)}.
$$

For $c\geq c_{\min}$,

$$
L_{\max}(t,c,\alpha)
\approx
\min\left\{
\left\lfloor\frac{t}{c}\right\rfloor,
\frac{-\log \alpha}{\epsilon(c)}
\right\}.
$$

The first term gives the linear time-depth frontier. The second term is an error ceiling.

If $c$ is a design choice rather than fixed in advance, the best usable depth is predicted by optimizing this tradeoff:

$$
L_{\max}(t,\alpha)
\approx
\max_{\substack{c\in\mathbb N_+\\ c\geq c_{\min}}}
\min\left\{
\left\lfloor\frac{t}{c}\right\rfloor,
\frac{-\log \alpha}{\epsilon(c)}
\right\}.
$$

Small $c$ fits more transitions into the same internal budget, but makes each transition less reliable. Large $c$ makes each transition more reliable, but leaves fewer transition slots. The optimal $c$ is where these two constraints balance.

![time allocation tradeoff](<Images/Temporal and Iterative Tasks/time_allocation_tradeoff.png>)

*Figure: schematic time-allocation tradeoff. Increasing $c$ improves one-step reliability but reduces the number of executable transitions.*

![accuracy vs time](<Images/Temporal and Iterative Tasks/accuracy_vs_time.png>)

*Figure: fixed-depth accuracy curves. Deeper chains require more internal time before execution is possible, and then remain lower because transition errors accumulate.*

## 13. Pointer-Chasing Predictions

The theory predicts these measurements.

| Measurement | Prediction |
|---|---|
| accuracy vs $t$ at fixed $L$ | low before $t\approx cL$, then rises toward an error-limited plateau |
| accuracy vs $L$ at fixed $c$ | approximately exponential decay when $\epsilon(c)>0$ |
| heatmap over $(L,t)$ | diagonal transition boundary near $t=cL$ |
| per-step state margin $m_\ell$ | margins weaken before decoded trajectory errors |
| one-step error $\epsilon(c)$ | falls with $c$ until an irreducible floor |
| usable depth vs $c$ | intermediate $c$ can be optimal because reliability and executable depth trade off |
| transition confusion matrix $P_c^M(v\mid s)$ | identifies directed crosstalk pairs |

The main thesis-level claim is:

$$
\text{iterative-task depth grows roughly linearly with available internal AC time, until error accumulation becomes dominant.}
$$

This is qualitatively different from static classification. Static tasks are predicted to rise and saturate with internal time unless drift occurs. Iterative tasks are predicted to show a time-depth frontier because each dependent transition consumes update budget.

The schematic plots in this note are generated by [generate_temporal_task_plots.py](<Plots/Temporal and Iterative Tasks/generate_temporal_task_plots.py>).
