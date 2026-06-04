**Purpose:** define the thesis taxonomy of “time” in AC, and separate the different roles internal time can play.

In these notes, $r$ is the internal update index and $t$ is the number of internal AC update steps allowed before readout:

$$
r=0,1,\dots,t.
$$

This note is an interpretation layer on top of the standard AC update rules. It fixes how the thesis will use the word “time”; later notes study how task performance changes as $t$ varies.

## 1. External Time

External time is task-level time: the order, length, or horizon specified by the task itself.

Examples:

- an MNIST image is presented to the model;
- a sequence of symbols is presented one symbol at a time;
- a pointer-chasing instance supplies a table $M$, start state $s_0$, and desired depth $L$.

If a task provides inputs

$$
q_0,q_1,\dots,q_L,
$$

then $L$ is the external task length. It is different from the internal time budget $t$.

For pointer chasing, $L$ is the requested transition depth. The $L$ successive transitions may then be executed internally by the model:

$$
\text{external task depth }L
\neq
\text{internal update budget }t.
$$

## 2. Internal Time

Internal time is the number of AC update rounds run after an input, cue, or current state is supplied.

For a fixed input $q$, the internal trajectory is

$$
X_q(0),X_q(1),\dots,X_q(t).
$$

The model output is read from $X_q(t)$.

Each internal update step computes synaptic input, applies the $k$-cap rule, and optionally applies plasticity if the experiment allows weights to change. During evaluation, all updates use the fixed learned weights $W^\star$ unless explicitly stated otherwise.

## 3. Stimulus Time and Non-Stimulus Time

There are two different ways an input can interact with internal time.

**Internal stimulus time:** the external drive remains present while the system updates.

$$
u_A(r)=u_A
\qquad r=0,\dots,t-1.
$$

This asks how the network evolves while the stimulus is still being applied.

**Internal non-stimulus time:** the input is used to initialise or cue the system, then the system evolves without continued external drive. The cue may be represented either by the initial state $X(0)$ or by a drive at the first step:

$$
X(0)\text{ encodes the cue}
\quad\text{or}\quad
u_A(0)\neq 0,
\qquad
u_A(r)=0\quad\text{for }r>0.
$$

This asks how the network settles, completes, or computes after the initial cue.

## 4. Time as Settling

In some tasks, additional internal time allows activity to move toward a stable or more useful state.

This is the weakest meaning of time. It does not imply that the model is carrying out a long algorithm. It only means the first update may not be the final useful state.

The expected effect is improvement for small $t$, followed by saturation once the activity has stabilised.

## 5. Time as Denoising or Completion

In pattern-completion tasks, time can help recover a learned assembly from a partial or noisy cue.

For example, a noisy or partial cue may initially activate the wrong neurons. Recurrent dynamics may then increase overlap with a learned assembly:

$$
|\operatorname{supp}(x_A(r))\cap S|
$$

where $S$ is the target assembly.

This is stronger than simple settling. The system is not only becoming stable; it is moving from an incomplete or corrupted representation toward a stored representation.

## 6. Time as Execution

In temporal or iterative tasks, time is not just settling. It is the computation itself.

For an iterated transition problem,

$$
s_{\ell+1}=F(s_\ell),
$$

each step depends on the result of the previous step. Pointer chasing has this form:

$$
s_{\ell+1}=M(s_\ell).
$$

In an AC implementation, one transition may require one internal update or a short block of internal updates. Let $c$ be the number of internal AC updates required per transition. Then after $t$ internal updates, the model can execute approximately

$$
\left\lfloor \frac{t}{c} \right\rfloor
$$

transitions.

This is the central reason pointer chasing is useful for the thesis: success depends on repeatedly applying the same transition rule, so increasing internal time can increase the achievable computation depth, subject to transition errors.

## 7. Time as Interference

More time is not automatically better.

Additional recurrent updates can cause activity to drift away from the desired assembly, especially when learned assemblies overlap or have similar inputs. In classification, this may cause the active cap to drift toward a competing class assembly, increasing overlap with the wrong class:

$$
|\operatorname{supp}(x_A(r))\cap S_{\mathrm{wrong}}|
$$

may increase with $r$.

This is one mechanism by which static tasks can degrade when given too much internal time. The same recurrent dynamics that help completion can also amplify overlap, ambiguity, or noise.

## 8. Time as a Resource

The thesis treats internal time as a resource separate from model size.

The fixed instantiated model is

$$
\theta_{\mathrm{inst}}
=
(\mathcal A,\mathcal F,G,W^\star,k).
$$

The internal time budget $t$ is varied while $\theta_{\mathrm{inst}}$ is held fixed. Performance is written as

$$
Acc_{\mathcal Q}(\theta_{\mathrm{inst}},t).
$$

For depth-indexed tasks such as pointer chasing, it is useful to write the task distribution as $\mathcal Q_L$, where $L$ is the required transition depth:

$$
Acc_{\mathcal Q_L}(\theta_{\mathrm{inst}},t).
$$

This allows two different questions to be separated:

$$
\text{For fixed }L,\text{ how does accuracy change with }t?
$$

and

$$
\text{For fixed }t,\text{ how large can }L\text{ be before accuracy fails?}
$$

This notation asks a precise question:

$$
\text{What changes when the same AC system is allowed more internal update steps?}
$$

That framing is different from asking whether AC is computationally powerful with unlimited time and space. The thesis question is about useful computation under fixed resources.

## 9. Summary

| Meaning of time | Role in the thesis |
|---|---|
| External time | task-level order, length, or horizon |
| Internal stimulus time | updates while input remains present |
| Internal non-stimulus time | updates after a cue or initial state |
| Settling time | movement toward a stable or more useful state |
| Denoising / completion time | recovery of a learned assembly from a partial cue |
| Execution time | repeated application of a learned transition |
| Interference time | opportunity for drift, overlap, or error amplification |
| Resource time | variable $t$ studied while $\theta_{\mathrm{inst}}$ is fixed |

The later theory notes use this distinction to explain why static tasks can saturate, why iterative tasks can require time, and why additional time can sometimes hurt performance.
