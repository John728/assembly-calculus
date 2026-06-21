**Purpose:** Derive how internal time should affect non-iterative tasks, especially static classification.

## 1. Static Tasks and Setup

### 1.1 Defining the Static Task Regime

A static task is a direct mapping:
$$
q \mapsto y,
$$
where the input $q$ is a fixed, non-sequential sensory pattern (or cue) and the target label $y$ is a single discrete class. 

### 1.2 Why MNIST is a Static Task
MNIST digit classification is a classic example of a static task. The input $q$ is a static pixel image (represented in Assembly Calculus as a fixed set of active input sensory neurons), and the target $y$ is a static class label (represented by a pre-formed target assembly $S_c$ in the coding area). 

Unlike sequential sequence tracking (such as DFA path tracking) or iterative memory traversal (such as pointer chasing), classifying an MNIST digit does not require the network to transition through a sequence of intermediate states. The computation has a temporal dependency depth of $L = 0$. Consequently, the role of internal recurrent time $t$ in a static task is not to process sequential transitions, but to act as a **settling resource** to resolve input noise, complete pattern features, or recruit the correct assembly representation under the network's recurrent dynamics.

### 1.3 Mathematical Setup

Let $\mathcal C=\{1,\dots,C\}$ be the set of classes. For each class $c\in\mathcal C$, assume training has produced a class assembly

$$
S_c\subseteq \{1,\dots,n\},
\qquad
|S_c|=k
$$

in a readout area $A$.

For an input $q$ with true class $y$, let

$$
K_q(r)=\operatorname{supp}(x_A(r))
$$

be the active cap in the readout area after $r$ internal updates.

Define the class-overlap score

$$
o_c(r)
=
\frac{|K_q(r)\cap S_c|}{k}.
$$

Thus $o_c(r)\in[0,1]$. A natural assembly readout is

$$
\hat y_t(q)
=
\arg\max_{c\in\mathcal C} o_c(t).
$$

The correct-class margin is

$$
m_y(t)
=
o_y(t)-\max_{c\neq y}o_c(t).
$$

The input is classified correctly whenever the true class has strictly positive margin:

$$
m_y(t)>0.
$$

If $m_y(t)=0$, correctness depends on the tie-breaking rule. Thus positive margin is the clean no-tie success condition.

For a static task distribution $\mathcal Q$,

$$
Acc_{\mathcal Q}(\theta_{\mathrm{inst}},t)
=
\Pr_{(q,y)\sim\mathcal Q}
\left[
m_y(t)>0
\right].
$$

## 2. Static Tasks Contain No Execution Depth

After the cue has been supplied, a static task has no required transition depth $L$. The target is already determined by $q$.

With evaluation weights fixed at $W^\star$, the post-cue AC dynamics can be written abstractly as

$$
X_q(r+1)=T(X_q(r);W^\star),
\qquad r=0,\dots,t-1.
$$

This is the non-stimulus-time case. If the stimulus is held present during evaluation, the update map also includes the constant external drive.

Therefore

$$
X_q(t)=T^t(X_q(0);W^\star).
$$

Internal time can change the representation, but it does not introduce new information about the label. For static tasks, time can help only if the recurrent dynamics move the state toward a representation with a larger classification margin.

If the trajectory reaches a fixed point after $\tau$ steps,

$$
X_q(\tau+1)=X_q(\tau),
$$

then for all $t\geq \tau$,

$$
\hat y_t(q)=\hat y_\tau(q).
$$

This gives the first prediction:

$$
\text{static-task performance is predicted to saturate unless later dynamics cause drift.}
$$

This is the point of contact with [[Temporal and Iterative Tasks]]. Both notes use the same AC ingredients: caps, overlaps, signal margins, and population $k$-cap thresholds. The difference is the outer computation being measured.

| Quantity | Static task | Temporal or iterative task |
|---|---|---|
| target | one label $y$ | final state $s_L=M^L(s_0)$ |
| main overlap | class overlap $o_c(t)$ | state overlap $o_s(r)$ |
| main margin | readout margin $m_y(t)$ | transition margin $D_{s,v}^{(c)}$ and state margin $m_\ell(r)$ |
| role of $t$ | settling, completion, or drift of one representation | enough updates to execute repeated dependent transitions |
| main constraint | $m_y(t)>0$ | $t\geq cL$ and $L\epsilon(c)\ll 1$ |

## 3. AC Signal Margin Under k-Cap

The overlap equations should be tied back to the actual AC update. In the readout area $A$, ignoring area labels for clarity, the input to neuron $i$ at step $r$ is

$$
I_i(r)
=
\sum_{j\in K_q(r)}
W_{ji}^\star
+u_i(r).
$$

Let $\tau_k(r)$ be the $k$-th largest value among all readout-area inputs:

$$
\tau_k(r)
=
\operatorname{kthlargest}\{I_1(r),\dots,I_n(r)\}.
$$

Then the next cap is

$$
K_q(r+1)
=
\{i:I_i(r)\geq \tau_k(r)\},
$$

up to the tie-breaking rule. Therefore the class overlap after one AC update is

$$
o_c(r+1)
=
\frac{1}{k}
\sum_{i\in S_c}
\mathbf 1\{I_i(r)\geq \tau_k(r)\}.
$$

This equation is exact apart from ties. It says that overlap grows when neurons in $S_c$ receive enough input to enter the next $k$-cap.

The threshold $\tau_k(r)$ is not an external constant. It is determined by the whole readout population. Since exactly $k$ out of $n$ neurons enter the cap,

$$
\frac{k}{n}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1\{I_i(r)\geq \tau_k(r)\},
$$

again up to ties. In a mean-field approximation, divide the readout population into groups $g\in\mathcal G$, such as the true assembly, competing assemblies, and background neurons. Let $\pi_g$ be the fraction of readout neurons in group $g$. Then $\tau_k(r)$ is approximately determined by

$$
\frac{k}{n}
\approx
\sum_{g\in\mathcal G}
\pi_g
\Pr[I^{(g)}(r)\geq \tau_k(r)].
$$

Under normal input approximations,

$$
\frac{k}{n}
\approx
\sum_{g\in\mathcal G}
\pi_g
\Phi\left(
\frac{\mu_g(r)-\tau_k(r)}
{\sigma_g(r)}
\right),
$$

where $\Phi$ is the standard normal CDF. This equation makes the threshold an AC population quantity: changing the input distribution of any large group can move $\tau_k(r)$ and therefore affect every class overlap.

![k-cap threshold condition](<Images/Static Tasks/kcap_threshold_condition.png>)

*Figure: schematic population threshold condition. The k-cap threshold is where the total population tail mass equals $k/n$.*

For a specific class $c$, approximate the inputs to neurons in $S_c$ as exchangeable random variables with mean $\mu_c(r)$ and standard deviation $\sigma_c(r)$. Then

$$
\mathbb E[o_c(r+1)]
\approx
\Pr[I^{(c)}(r)\geq \tau_k(r)].
$$

Under a normal approximation,

$$
\mathbb E[o_c(r+1)]
\approx
\Phi\left(
\frac{\mu_c(r)-\tau_k(r)}
{\sigma_c(r)}
\right),
$$

This gives an AC-level signal margin:

$$
\gamma_c(r)=\mu_c(r)-\tau_k(r).
$$

Increasing $\gamma_y(r)$ increases the expected overlap with the true assembly. Increasing $\gamma_z(r)$ for a competing class increases the expected wrong-class overlap.

Now approximate the mean input to a neuron in $S_c$. Let $w_{d\to c}$ be the all-pairs average learned recurrent weight from active neurons belonging to assembly $S_d$ into neurons of assembly $S_c$, with absent edges counted as zero. Let $w_0$ be the corresponding all-pairs background recurrent weight. If the current cap has overlap $o_d(r)$ with $S_d$, then approximately $k o_d(r)$ active presynaptic neurons come from $S_d$. This gives the multiclass crosstalk model

$$
\mu_c(r)
\approx
k\left[
w_0+
\sum_{d\in\mathcal C}
o_d(r)\left(w_{d\to c}-w_0\right)
\right]
+u_c(r).
$$

The diagonal term $w_{c\to c}$ supports recall of assembly $S_c$. Off-diagonal terms $w_{d\to c}$ are synaptic crosstalk from other assemblies into $S_c$. There is no separate factor of $p$ because the averages already include missing synapses as zero.

This treats the overlaps $o_d(r)$ as first-order explanatory variables. If assemblies overlap substantially, the sum over $d$ is not an exact partition of the cap; it is a linear mean-field approximation to the recurrent input.

Thus the AC update suggests a multiclass mean-field overlap map

$$
\mathbb E[o_c(r+1)]
\approx
\Phi\left(
\frac{
k\left[
w_0+
\sum_{d\in\mathcal C}
o_d(r)\left(w_{d\to c}-w_0\right)
\right]
+u_c(r)
-\tau_k(r)
}
{\sigma_c(r)}
\right).
$$

Write this compactly as

$$
o_c(r+1)\approx F_c(o(r)).
$$

This is the bridge between the raw AC rule and the lower-dimensional overlap models used below. A learned assembly is stable when this mean-field map has a high-overlap fixed point and the local slope around that fixed point has magnitude less than one.

![k-cap signal margin](<Images/Static Tasks/kcap_signal_margin.png>)

*Figure: schematic mean-field relation between AC signal margin and expected class overlap after one k-cap update.*

## 4. Settling Model

Suppose the input is in the basin of the correct class assembly $S_y$. The AC signal-margin map above motivates modelling the correct overlap as moving toward a stable value $o_y^\star$, where

$$
0<o_y^\star\leq 1.
$$

This allows partial recall: the limiting overlap need not be perfect.

Let $F_y$ be the mean-field overlap map from Section 3, restricted to the correct-class coordinate near the fixed point. A first-order expansion gives

$$
F_y(o_y)
\approx
o_y^\star+\rho_y(o_y-o_y^\star),
$$

where

$$
\rho_y=F_y'(o_y^\star).
$$

For a stable local attractor, $|\rho_y|<1$. In the monotone settling case used here, assume $0<\rho_y<1$.

Define the missing-overlap error

$$
d_y(r)=o_y^\star-o_y(r).
$$

A simple settling assumption is local contraction around the stable overlap:

$$
d_y(r+1)\leq \rho_y d_y(r),
\qquad
0<\rho_y<1.
$$

Then

$$
d_y(t)\leq \rho_y^t d_y(0),
$$

and therefore

$$
o_y(t)
\geq
o_y^\star-\left(o_y^\star-o_y(0)\right)\rho_y^t.
$$

This predicts fast early improvement and then diminishing returns. If the strongest wrong-class overlap is bounded by $b<1$, then correct classification is guaranteed once

$$
o_y(t)>b.
$$

Using the lower bound above, a sufficient settling time is

$$
t_{\mathrm{settle}}
=
\left\lceil
\frac{
\log\left(\frac{o_y^\star-b}{o_y^\star-o_y(0)}\right)
}{
\log \rho_y
}
\right\rceil,
$$

when $o_y(0)<b<o_y^\star$ and $0<\rho_y<1$.

This is the mathematical version of the settling-time hypothesis: early internal updates can improve accuracy because overlap with the correct assembly grows approximately geometrically.

![Settling overlap](<Images/Static Tasks/settling_overlap.png>)

*Figure: schematic settling curve from the contraction model. Correct overlap rises quickly, crosses the strongest wrong-class overlap, and then saturates.*

## 5. Competing-Assembly Model

Static tasks can also get worse with time. This happens when recurrent dynamics strengthen a competing class assembly as well as, or more than, the correct one.

Let $z$ be the strongest competing class for an input of true class $y$. Model the true and competing overlaps by

$$
o_y(r+1)
=
o_y(r)+\lambda_y\left(o_y^\star-o_y(r)\right),
$$

and

$$
o_z(r+1)
=
o_z(r)+\lambda_z\left(o_z^\star-o_z(r)\right),
$$

where $0<\lambda_y,\lambda_z<1$.

This is a phenomenological two-overlap projection of the full $k$-cap dynamics. It should be read as a low-dimensional fitted model, not as an exact theorem derived from the raw AC update. The effects of the cap constraint, background neurons, all other classes, and the AC signal margins $\gamma_y,\gamma_z$ are absorbed into the fitted quantities $o_y^\star$, $o_z^\star$, $\lambda_y$, and $\lambda_z$.

Equivalently, these recurrences are the linearized form of the mean-field map

$$
o_c(r+1)\approx F_c(o(r))
$$

near the relevant fixed point. In the one-dimensional projection,

$$
F_c(o_c)
\approx
o_c^\star+\rho_c(o_c-o_c^\star),
$$

with

$$
\rho_c=F_c'(o_c^\star),
\qquad
\lambda_c=1-\rho_c.
$$

The symbols are:

| Symbol | Meaning |
|---|---|
| $o_y^\star$ | long-run overlap with the true assembly |
| $o_z^\star$ | long-run overlap with the competing assembly |
| $\lambda_y$ | rate of attraction toward the true assembly |
| $\lambda_z$ | rate of attraction toward the competing assembly |

Let

$$
\rho_y=1-\lambda_y,
\qquad
\rho_z=1-\lambda_z.
$$

Solving the recurrences gives

$$
o_y(t)
=
o_y^\star-\left(o_y^\star-o_y(0)\right)\rho_y^t,
$$

and

$$
o_z(t)
=
o_z^\star-\left(o_z^\star-o_z(0)\right)\rho_z^t.
$$

Define

$$
\Delta_y=o_y^\star-o_y(0),
\qquad
\Delta_z=o_z^\star-o_z(0),
$$

and the pairwise margin

$$
m_{yz}(t)=o_y(t)-o_z(t).
$$

Then

$$
m_{yz}(t)
=
m_{yz}^\star
-\Delta_y\rho_y^t
+\Delta_z\rho_z^t,
$$

where

$$
m_{yz}^\star=o_y^\star-o_z^\star.
$$

This single equation gives the main static-task prediction. In the full multiclass setting, the strongest wrong class may change with time:

$$
z(t)=\arg\max_{c\neq y}o_c(t).
$$

The pairwise formulas apply on intervals where the same competitor $z$ remains the strongest wrong class.

![Competing overlap and margin](<Images/Static Tasks/competing_overlap_margin.png>)

*Figure: schematic competing-assembly trajectory. The competitor grows more slowly at first but has higher long-run support, eventually forcing the margin below zero.*

## 6. Rise, Plateau, or Fall

The margin changes according to

$$
m_{yz}(t+1)-m_{yz}(t)
=
\Delta_y(1-\rho_y)\rho_y^t
-
\Delta_z(1-\rho_z)\rho_z^t.
$$

There are three regimes.

### 6.1 Rise and Plateau

If the true assembly has a stronger asymptote and stronger or comparable attraction,

$$
m_{yz}^\star>0
$$

and

$$
\Delta_y(1-\rho_y)\rho_y^t
\geq
\Delta_z(1-\rho_z)\rho_z^t
$$

for the relevant range of $t$, then the margin rises or remains positive. The model predicts early improvement followed by a plateau.

This is expected for well-separated classes.

### 6.2 Saturation

If both overlaps have stabilised, then

$$
\rho_y^t\approx 0,
\qquad
\rho_z^t\approx 0,
$$

so

$$
m_{yz}(t)\approx m_{yz}^\star.
$$

Accuracy is predicted to stop changing once most examples have reached their limiting margins.

### 6.3 Fall or Inverted-U Behaviour

If the competing assembly has enough long-run support, then the margin can decrease after some number of recurrent updates.

A vulnerable pair satisfies

$$
m_{yz}(0)>0
$$

but

$$
m_{yz}^\star\leq 0.
$$

Then the input may be initially classified correctly but eventually drift toward the competing class.

If $\rho_y\approx\rho_z=\rho$, then

$$
m_{yz}(t)
\approx
m_{yz}^\star
+
\left(m_{yz}(0)-m_{yz}^\star\right)\rho^t.
$$

When $m_{yz}(0)>0$ and $m_{yz}^\star<0$, the approximate crossing time is

$$
t_{\mathrm{cross}}
\approx
\left\lceil
\frac{
\log\left(\frac{-m_{yz}^\star}{m_{yz}(0)-m_{yz}^\star}\right)
}{
\log\rho
}
\right\rceil.
$$

For $t<t_{\mathrm{cross}}$, the input is predicted to classify correctly. For $t>t_{\mathrm{cross}}$, it is predicted to classify as the competitor.

In the more general case $\rho_y\neq\rho_z$, the margin peak occurs near the solution of

$$
\Delta_y(1-\rho_y)\rho_y^t
=
\Delta_z(1-\rho_z)\rho_z^t.
$$

Solving gives

$$
t_{\mathrm{peak}}
\approx
\frac{
\log\left(
\frac{\Delta_z(1-\rho_z)}
{\Delta_y(1-\rho_y)}
\right)
}{
\log\left(\frac{\rho_y}{\rho_z}\right)
},
$$

when $\Delta_y,\Delta_z>0$ and the expression is positive and finite. This predicts inverted-U behaviour: time first helps, then hurts.

*Figure: three possible margin regimes produced by the same two-overlap model: rise-and-plateau, fall after drift, and inverted-U behaviour.*

### 6.4 Empirical Stimulus Regimes: Held vs. Transient

In practical evaluations (such as the MNIST classification experiments in [[Experimental-Validation]]), the stimulus presentation protocol determines which regime dominates:

1. **Held Stimulus (Stable Attractor):** When the input cue is held active across all updates, it acts as a constant feedforward drive. This drive stabilizes the target assembly's attractor basin, preventing the theoretical "drift" or "inverted-U" degradation. Empirically, accuracy rises early (from $69.0\%$ to $70.5\%$) and remains perfectly flat and stable all the way to $t=100$.
2. **Transient Stimulus (Attractor Collapse):** When the cue is only presented at $t=0$ and then removed, the network must rely purely on recurrent feedback. Because recurrent self-support is weak and homeostatic class-separation bias is high, the system does not gradually drift to a competitor; rather, the correct representation suffers a rapid **attractor collapse** to the noise floor (accuracy dropping to $10.0\%$ by $t=10$, with correct overlap collapsing to $0.09$).


## 7. Why Similar Classes Degrade First

Two class assemblies interfere when the active cap receives substantial support from both assemblies.

Define the normalised assembly overlap

$$
\Omega_{yz}
=
\frac{|S_y\cap S_z|}{k}.
$$

If $S_y$ and $S_z$ were independent random $k$-subsets of an area of size $n$, then

$$
\mathbb E[|S_y\cap S_z|]
=
\frac{k^2}{n},
$$

and

$$
\mathbb E[\Omega_{yz}]
=
\frac{k}{n}.
$$

For the MNIST hyperparameters used in Thesis A, $n=1000$ and $k=100$, so the random-overlap baseline is

$$
\mathbb E[\Omega_{yz}]=0.1.
$$

Real trained assemblies are not purely independent random sets. Similar classes can have larger effective overlap because they share sensory features and training history.

Physical overlap is only one source of interference. Similar classes can also interfere through learned synaptic crosstalk even when $S_y\cap S_z$ is small. Define the directed average crosstalk from assembly $S_d$ into assembly $S_c$ as

$$
C_{d\to c}
=
\frac{1}{k^2}
\sum_{j\in S_d}
\sum_{i\in S_c}
W_{ji}^\star.
$$

Absent edges contribute $W_{ji}^\star=0$ in this average. Large $C_{y\to z}$ means activity in the true assembly $S_y$ can directly help drive neurons in the competing assembly $S_z$ into the cap.

Let $h_z(q)$ denote the initial feedforward evidence for competitor $z$ on input $q$. The competitor risk increases with feedforward evidence, structural overlap, and directed crosstalk. A simple way to record this is an interference score

$$
I_{yz}(q)
=
h_z(q)+\alpha\Omega_{yz}+\eta C_{y\to z},
\qquad
\alpha,\eta\geq 0.
$$

Large $I_{yz}(q)$ predicts larger $o_z^\star$ or faster competitor growth. Therefore visually similar class pairs are predicted to degrade before well-separated class pairs.

For MNIST, this matches the observed qualitative pattern: classes such as $9$ and $7$, or $5$ and $3$, are more likely to show increasing wrong-class overlap as $t$ grows.

## 8. From Margin to Accuracy

For class $y$, define the per-class accuracy

$$
Acc_y(t)
=
\Pr[m_y(t)>0\mid y].
$$

Overall accuracy is

$$
Acc_{\mathcal Q}(\theta_{\mathrm{inst}},t)
=
\sum_{y\in\mathcal C}
\pi_y Acc_y(t),
$$

where $\pi_y=\Pr[y]$.

The most direct empirical estimate uses the margin CDF. Let

$$
\widehat F_{m_y(t)\mid y}(a)
=
\widehat{\Pr}[m_y(t)\leq a\mid y].
$$

Then the positive-margin accuracy is

$$
Acc_y(t)
\approx
1-\widehat F_{m_y(t)\mid y}(0).
$$

This is preferable experimentally because margins are bounded and discrete in steps of $1/k$.

If margins within a class are approximately normal, this can be simplified. Suppose

$$
m_y(t)\sim \mathcal N(\mu_y(t),\sigma_y^2(t)),
$$

then

$$
Acc_y(t)
\approx
\Phi\left(
\frac{\mu_y(t)}{\sigma_y(t)}
\right),
$$

where $\Phi$ is the standard normal CDF.

This connects the overlap model to plots: if the mean margin $\mu_y(t)$ rises and saturates, accuracy rises and saturates. If $\mu_y(t)$ falls because a competitor grows, accuracy falls.

Mean margin alone is not sufficient unless the variance or lower tail is controlled. In experiments, the margin spread should also be measured:

$$
\sigma_y(t)
$$

or, more robustly, lower margin quantiles such as

$$
Q_{0.1}[m_y(t)\mid y].
$$

A rising mean with a worsening lower tail can still produce falling accuracy.

![Accuracy from margin](<Images/Static Tasks/accuracy_from_margin.png>)

*Figure: mapping from mean margin to expected accuracy. Classes with falling margins are predicted to lose accuracy as internal time increases.*

## 9. Hyperparameter Predictions

The model predicts the following qualitative effects.

| Parameter change | Predicted static-task effect |
|---|---|
| larger $n$ with fixed $k$ | lower random overlap $k/n$, less interference |
| larger $k/n$ | more shared neurons between assemblies, higher interference risk |
| larger $p$ | stronger recurrent signal, but also stronger crosstalk |
| larger $\beta$ during training | stronger assemblies and faster settling, but possible faster drift to wrong attractors |
| larger $t$ | useful until settling; harmful if competing margins overtake |

The important point is that $t$ is not predicted to be uniformly good or bad. Its effect depends on whether the correct margin or the competing margin grows faster.

## 10. Experimental Signatures

The theory predicts the following plots.

| Plot | Predicted signature |
|---|---|
| $o_y(t)$ vs $t$ | correct overlap rises then plateaus for well-separated inputs |
| $\max_{z\neq y}o_z(t)$ vs $t$ | wrong overlap remains low for easy classes, rises for confusing classes |
| $m_y(t)$ vs $t$ | margin rises, saturates, or crosses zero |
| empirical CDF of $m_y(t)$ | directly estimates positive-margin accuracy |
| $\sigma_y(t)$ or margin quantiles vs $t$ | reveals whether tail examples degrade despite mean improvement |
| $Acc_y(t)$ vs $t$ | per-class accuracy reveals which classes degrade |
| confusion matrix vs $t$ | errors concentrate in high-interference pairs |

For static classification, the strongest prediction is:

$$
\text{extra internal time helps only while it increases the correct-class margin.}
$$

Once recurrent dynamics stop increasing that margin, accuracy is predicted to saturate. If recurrent dynamics increase a competing-class margin faster, accuracy is predicted to fall.

The schematic plots in this note are generated by [generate_static_task_plots.py](<Plots/Static Tasks/generate_static_task_plots.py>).
