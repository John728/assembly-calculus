**Purpose:** explain why small local errors can dominate long temporal computations, and separate no-error trajectory accuracy from final-state accuracy.

This note supports [[Temporal and Iterative Tasks]] and [[Time–Size Tradeoff]]. The key idea is simple: internal time can make each transition more reliable, but a depth-$L$ computation creates $L$ opportunities for failure.

## 1. One-Step Error

Consider a transition task with true update rule

$$
s_{\ell+1}=M(s_\ell).
$$

Let $\widehat M_c(s)$ be the state decoded after giving the AC transition mechanism $c$ internal updates from current state $s$.

The one-step error at state $s$ is

$$
\epsilon_s(c)
=
\Pr[\widehat M_c(s)\neq M(s)].
$$

If states are sampled from a distribution $\nu$, define

$$
\epsilon(c)
=
\mathbb E_{s\sim\nu}\epsilon_s(c).
$$

In AC terms, $\epsilon_s(c)$ is controlled by the post-transition state margin. For a wrong destination $v\neq M(s)$, define

$$
D_{s,v}^{(c)}
=
o_{M(s)}(r+c)-o_v(r+c).
$$

The decoded transition is correct when

$$
\min_{v\neq M(s)}D_{s,v}^{(c)}>0.
$$

Thus

$$
\epsilon_s(c)
=
\Pr\left[
\min_{v\neq M(s)}D_{s,v}^{(c)}\leq 0
\right].
$$

If the pairwise margin $D_{s,v}^{(c)}$ has mean $\delta_{s,v}(c)$ and standard deviation $\eta_{s,v}(c)$, then

$$
\Pr[D_{s,v}^{(c)}\leq 0]
\approx
\Phi\left(
-\frac{\delta_{s,v}(c)}
{\eta_{s,v}(c)}
\right).
$$

By a union bound over wrong destinations,

$$
\epsilon_s(c)
\leq
\sum_{v\neq M(s)}
\Phi\left(
-\frac{\delta_{s,v}(c)}
{\eta_{s,v}(c)}
\right).
$$

This is the AC link: weak destination margins create local transition errors, and local transition errors accumulate over depth.

## 2. Error as a Function of Per-Transition Time

A useful model is

$$
\epsilon(c)
\approx
\epsilon_\infty+\epsilon_0\rho^c,
\qquad
0<\rho<1.
$$

Here $\epsilon_\infty$ is the irreducible error floor from finite capacity, crosstalk, table ambiguity, or noise. The term $\epsilon_0\rho^c$ is the part reduced by allowing more AC updates per transition.

Therefore increasing $c$ helps only until

$$
\epsilon(c)\approx \epsilon_\infty.
$$

If $\epsilon_\infty>0$, arbitrarily long path-correct chains still eventually fail with high probability in the no-recovery approximation.

## 3. Multi-Step No-Error Probability

Let $E_\ell$ be the event that transition $\ell$ is correct, conditioned on all previous transitions being correct.

The no-error trajectory event is

$$
E_0\cap E_1\cap\cdots\cap E_{L-1}.
$$

By the chain rule,

$$
\Pr[\text{no transition error}]
=
\prod_{\ell=0}^{L-1}
\Pr[E_\ell\mid E_0,\dots,E_{\ell-1}].
$$

Let the conditional transition error be

$$
\epsilon_\ell
=
1-\Pr[E_\ell\mid E_0,\dots,E_{\ell-1}].
$$

Then

$$
\Pr[\text{no transition error}]
=
\prod_{\ell=0}^{L-1}
(1-\epsilon_\ell).
$$

If every conditional error is bounded by $\epsilon$, then

$$
\Pr[\text{no transition error}]
\geq
(1-\epsilon)^L.
$$

For small $\epsilon$,

$$
(1-\epsilon)^L
\approx
e^{-\epsilon L}.
$$

More generally, define the accumulated hazard

$$
H_L
=
\sum_{\ell=0}^{L-1}\epsilon_\ell.
$$

When all $\epsilon_\ell$ are small,

$$
\prod_{\ell=0}^{L-1}(1-\epsilon_\ell)
\approx
e^{-H_L}.
$$

![success vs depth](<Images/Error Accumulation/success_vs_depth.png>)

*Figure: no-error trajectory probability decays with depth. Small one-step errors remain hidden at short depth but dominate long chains.*

## 4. Union-Bound Failure Control

The probability of at least one transition error is

$$
\Pr[\text{some transition error}]
=
\Pr\left[
\bigcup_{\ell=0}^{L-1}E_\ell^c
\right].
$$

The union bound gives

$$
\Pr[\text{some transition error}]
\leq
\sum_{\ell=0}^{L-1}\epsilon_\ell.
$$

If $\epsilon_\ell\leq \epsilon$ for all $\ell$, then

$$
\Pr[\text{some transition error}]
\leq
L\epsilon.
$$

Thus a sufficient condition for high path correctness is

$$
L\epsilon\ll 1.
$$

For target path-correctness probability $\alpha$, a useful approximation is

$$
(1-\epsilon)^L\geq \alpha.
$$

Solving gives

$$
L
\leq
\frac{\log \alpha}{\log(1-\epsilon)}
\approx
\frac{-\log \alpha}{\epsilon}.
$$

This is the error ceiling used in the temporal and time-size notes.

## 5. Fixed Total Time and Depth

If the total internal budget is $t$ and it is spread across $L$ transitions, then

$$
c(t,L)
=
\left\lfloor\frac{t}{L}\right\rfloor.
$$

Let $c_{\min}$ be the minimum number of AC updates needed to perform one symbolic transition attempt. The approximation below applies in the executable regime

$$
c(t,L)\geq c_{\min}.
$$

If

$$
c(t,L)<c_{\min},
$$

then the model does not have enough internal budget to execute the requested $L$ transitions. Accuracy should be treated as baseline, early-stop accuracy, or overshoot/stop-control accuracy depending on the evaluation protocol.

Substituting into the one-step error model gives

$$
\epsilon(t,L)
\approx
\epsilon_\infty+\epsilon_0\rho^{\lfloor t/L\rfloor}.
$$

The path-correctness approximation becomes

$$
Acc_{\mathrm{path}}(L,t)
\approx
\left(1-\epsilon(t,L)\right)^L.
$$

Equivalently, for small errors,

$$
Acc_{\mathrm{path}}(L,t)
\approx
\exp\left(
-L\left[
\epsilon_\infty+\epsilon_0\rho^{\lfloor t/L\rfloor}
\right]
\right).
$$

This formula has two failure mechanisms:

- larger $L$ creates more factors in the product;
- larger $L$ also lowers $c(t,L)$ when $t$ is fixed, making each transition less reliable.

![fixed time depth](<Images/Error Accumulation/fixed_time_depth.png>)

*Figure: fixed total time creates a depth cliff. Deeper chains both accumulate more errors and receive fewer updates per transition.*

## 6. First-Error Position

If the per-step error is approximately constant at $\epsilon$, then the probability that the first error occurs exactly at transition $\ell$ is

$$
\Pr[T_{\mathrm{first}}=\ell]
=
(1-\epsilon)^\ell\epsilon,
\qquad
\ell=0,\dots,L-1.
$$

The probability of no error before depth $L$ is

$$
\Pr[T_{\mathrm{first}}\geq L]
=
(1-\epsilon)^L.
$$

The expected first-error time in an infinite chain is

$$
\mathbb E[T_{\mathrm{first}}]
=
\frac{1-\epsilon}{\epsilon}
$$

when $\ell$ is counted from $0$. This gives a direct experimental diagnostic: if first errors occur much earlier than predicted by a constant-$\epsilon$ model, transition reliability is worsening with depth or state distribution.

![first error distribution](<Images/Error Accumulation/first_error_distribution.png>)

*Figure: geometric first-error distribution. Long chains can fail even when early transitions usually look correct.*

## 7. Final-State Accuracy Versus Path Accuracy

No-error path accuracy is not always the same as final-state accuracy. The no-error event implies success:

$$
\{\text{no transition error}\}
\subseteq
\{\hat s_L=s_L\}.
$$

Therefore

$$
\Pr[\text{no transition error}]
\leq
\Pr[\hat s_L=s_L].
$$

The inequality can be strict when an incorrect intermediate state later collides with the correct trajectory or when the task has multiple states with the same acceptable output.

A compact recovery model tracks whether the trajectory is correct or wrong. Let $p_\ell$ be the probability that the decoded state is correct after $\ell$ transitions. Suppose

$$
\Pr[\text{correct}\to\text{wrong}]=\epsilon
$$

and

$$
\Pr[\text{wrong}\to\text{correct}]=r.
$$

Then

$$
p_{\ell+1}
=
p_\ell(1-\epsilon)
+
(1-p_\ell)r.
$$

When $r=0$,

$$
p_L=(1-\epsilon)^L.
$$

When $r>0$, final-state accuracy can be higher than no-error path accuracy. For path-novel pointer chasing, $r$ should be small because wrong states usually follow different future pointers.

![recovery comparison](<Images/Error Accumulation/recovery_comparison.png>)

*Figure: recovery can make final-state accuracy exceed no-error path accuracy. Path-novel pointer chasing is closer to the no-recovery case.*

## 8. Relation to Static Tasks

Static tasks do not have a required transition depth $L$. They can still accumulate errors over internal updates, but the mechanism is different.

For static classification, the relevant quantity is the class margin

$$
m_y(t)
=
o_y(t)-\max_{z\neq y}o_z(t).
$$

Extra time helps when this margin increases. Extra time hurts when recurrent dynamics push the cap toward a wrong attractor. This is drift, not required multi-step execution.

For temporal tasks, even if each transition is locally good, the computation faces the multiplicative constraint

$$
L\epsilon(c)\ll 1.
$$

That is why long pointer chains are a stronger test of internal sequential reliability than static classification.

## 9. Experimental Signatures

The theory predicts these measurements.

| Measurement                              | Prediction                                                     |
| ---------------------------------------- | -------------------------------------------------------------- |
| accuracy vs $L$ at fixed $c$             | approximately exponential decay when $\epsilon(c)>0$           |
| accuracy vs $t$ at fixed $L$             | rises as $c(t,L)$ increases, then plateaus at the error floor  |
| first-error position histogram           | geometric shape if per-step error is stable                    |
| per-step margin $D_{s,v}^{(c)}$          | lower margins predict higher first-error probability           |
| final-state accuracy minus path accuracy | estimates recovery, collisions, or many-to-one outputs         |
| confusion matrix by step                 | identifies whether errors are state-specific or depth-specific |

The main claim is:

$$
\text{long temporal computations are limited by accumulated local transition error, not only by one-step accuracy.}
$$

The schematic plots in this note are generated by [generate_error_accumulation_plots.py](<Plots/Error Accumulation/generate_error_accumulation_plots.py>).
