**Purpose:** fix the notation for the AC system used in the theory notes.

Use $r$ for the internal update index. Use $t$ for the number of internal update steps allowed before readout.

$$
r=0,1,\dots,t
$$

The output of a run is read from the state at $r=t$.

## 1. Areas and Activity

Let

$$
\mathcal A=\{A_1,\dots,A_a\}
$$

be the set of brain areas. Each area contains $n$ excitatory neurons.

The activity of area $A$ at internal step $r$ is

$$
x_A(r)\in\{0,1\}^n.
$$

The entry $x_{A,i}(r)=1$ means neuron $i$ fires at step $r$. In an active, uninhibited area the cap size is fixed:

$$
\|x_A(r)\|_0=k,
\qquad k\ll n.
$$

If an area is inhibited, it has no active excitatory neurons. Thus, in general,

$$
\|x_A(r)\|_0\in\{0,k\}.
$$

The active population is

$$
\operatorname{supp}(x_A(r))=\{i:x_{A,i}(r)=1\}.
$$

A $k$-cap is the active $k$-neuron set at one step. An assembly is a learned, stable $k$-sparse population that can be reliably recalled. In equations, assemblies are represented by their activity vectors. The full network state is

$$
X(r)=\bigl(x_{A_1}(r),\dots,x_{A_a}(r)\bigr).
$$

## 2. Fibres and Weights

Let

$$
\mathcal F\subseteq \mathcal A\times\mathcal A
$$

be the set of directed fibre systems in the architecture. If $(B,A)\in\mathcal F$, then area $B$ can send activity to area $A$.

The weight matrix for the fibre $B\to A$ is

$$
W^{B\to A}(r)\in\mathbb R_{\ge 0}^{n\times n},
$$

where $W^{B\to A}_{ji}(r)$ is the weight from neuron $j$ in $B$ to neuron $i$ in $A$.

Sparse connectivity is represented by a binary graph mask

$$
G^{B\to A}_{ji}\sim\operatorname{Bernoulli}(p).
$$

Absent edges have zero weight:

$$
G^{B\to A}_{ji}=0
\quad\Rightarrow\quad
W^{B\to A}_{ji}(r)=0.
$$

Existing edges are usually initialised as

$$
G^{B\to A}_{ji}=1
\quad\Rightarrow\quad
W^{B\to A}_{ji}(0)=1.
$$

The case $B=A$ gives recurrent connections inside an area. The case $B\neq A$ gives long-range fibres between areas.

## 3. Input and State Update

The total input to neuron $i$ in area $A$ is

$$
I^A_i(r)
=
\sum_{B:(B,A)\in\mathcal F}
\sum_{j=1}^{n}
W^{B\to A}_{ji}(r)x_{B,j}(r)
+u_{A,i}(r),
$$

where $u_A(r)\in\mathbb R_{\ge 0}^n$ is external drive.

Equivalently,

$$
I^A(r)
=
\sum_{B:(B,A)\in\mathcal F}
\bigl(W^{B\to A}(r)\bigr)^\top x_B(r)
+u_A(r).
$$

The $k$-cap operation keeps the $k$ largest entries:

$$
\operatorname{kcap}(z,k)\in\{0,1\}^n.
$$

For an active, uninhibited area and $r=0,\dots,t-1$,

$$
x_A(r+1)
=
\operatorname{kcap}\bigl(I^A(r),k\bigr).
$$

If gating is needed, let $g_A(r)\in\{0,1\}$. Then, for $r=0,\dots,t-1$,

$$
x_A(r+1)
=
\begin{cases}
\operatorname{kcap}\bigl(I^A(r),k\bigr), & g_A(r)=1,\\
0, & g_A(r)=0.
\end{cases}
$$

## 4. Hebbian Plasticity

For an existing edge from neuron $j$ in area $B$ to neuron $i$ in area $A$, the Hebbian update is

$$
W^{B\to A}_{ji}(r+1)
=
W^{B\to A}_{ji}(r)
\left(1+\beta x_{B,j}(r)x_{A,i}(r+1)\right).
$$

The parameter $\beta>0$ is the plasticity rate. If the source neuron fires at step $r$ and the target neuron fires at step $r+1$, the existing weight is multiplied by $1+\beta$. Otherwise it is unchanged.

Absent edges remain absent:

$$
G^{B\to A}_{ji}=0
\quad\Rightarrow\quad
W^{B\to A}_{ji}(r)=0
\quad\text{for all }r.
$$

If weight normalisation is used, it is treated as part of the model specification rather than a separate computational resource.

## 5. Fixed Resources and Time

It is useful to distinguish a random model family from one realised model. The family-level resources are

$$
\theta_{\mathrm{family}}
=
(\mathcal A,\mathcal F,n,k,p,\beta).
$$

For one pre-training instantiation, the realised graph masks and initial weights also matter:

$$
\theta_{0}
=
(\mathcal A,\mathcal F,G,W(0),k,\beta).
$$

Here $G$ denotes all realised binary fibre masks. After training, evaluation uses fixed learned weights $W^\star$. Write the trained instantiated model as

$$
\theta_{\mathrm{inst}}
=
(\mathcal A,\mathcal F,G,W^\star,k).
$$

Experiments condition on $\theta_{\mathrm{inst}}$. Mean-field predictions may average over the graph distribution specified by $\theta_{\mathrm{family}}$ or over multiple trained instantiations.

The internal time budget is not included in the fixed resource descriptions:

$$
t\notin\theta_{\mathrm{family}},
\qquad
t\notin\theta_{0},
\qquad
t\notin\theta_{\mathrm{inst}}.
$$

This makes it possible to vary $t$ while holding the instantiated model, model size, and learning parameters fixed.

The number of areas is $a=|\mathcal A|$.

A simple neuron-count size measure is

$$
S_{\mathrm{neurons}}(\theta_{\mathrm{family}})=an.
$$

A simple expected synapse-count measure for the family is

$$
S_{\mathrm{synapses}}(\theta_{\mathrm{family}})
\approx
|\mathcal F|pn^2.
$$

For a realised model, the actual synapse count is

$$
S_{\mathrm{synapses}}(\theta_{\mathrm{inst}})
=
\sum_{(B,A)\in\mathcal F}
\sum_{j=1}^{n}
\sum_{i=1}^{n}
G^{B\to A}_{ji}.
$$

## 6. Readout and Performance

After training, write the learned weights as $W^\star$. During evaluation, weights are held fixed at $W^\star$ unless otherwise stated. For an input $q$, the evaluation trajectory is

$$
X_q(0),X_q(1),\dots,X_q(t).
$$

A task-specific readout map

$$
R:\prod_{A\in\mathcal A}\{0,1\}^n\to\mathcal Y
$$

produces

$$
\hat y_t(q)=R\bigl(X_q(t)\bigr).
$$

For a task distribution $\mathcal Q$, define performance as

$$
Acc_{\mathcal Q}(\theta_{\mathrm{inst}},t)
=
\Pr_{(q,y)\sim\mathcal Q}
\left[
\hat y_t(q)=y
\right].
$$

The main object studied later is how $Acc_{\mathcal Q}(\theta_{\mathrm{inst}},t)$ changes as $t$ varies with $\theta_{\mathrm{inst}}$ fixed. Later notes may write $\theta$ when the family/instance distinction is not central.
