from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from pyac.core.connectivity import build_connectivity
from pyac.core.dynamics import FeedforwardStrategy, RecurrentStrategy, RefractedStrategy
from pyac.core.inhibition import InhibitionState, k_cap
from pyac.core.plasticity import hebbian_update
from pyac.core.types import Assembly, AreaSpec, NetworkSpec
from pyac.core.weights import init_weights, normalize_weights


@dataclass
class StepResult:
    assemblies: dict[str, Assembly]
    step: int


@dataclass
class Trace:
    steps: list[StepResult]


class Network:
    def __init__(self, spec: NetworkSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng

        self.areas_by_name: dict[str, AreaSpec] = {area.name: area for area in spec.areas}
        self.area_names: list[str] = [area.name for area in spec.areas]
        self.step_order: list[str] = self._resolve_step_order(spec.step_order)

        self.connectivity: dict[tuple[str, str], csr_matrix] = build_connectivity(spec, rng)
        self.weights: dict[tuple[str, str], csr_matrix] = init_weights(self.connectivity)

        for area in spec.areas:
            normalize_weights(self.weights, area.name, spec)

        self.incoming_fibers: dict[str, list[str]] = {name: [] for name in self.area_names}
        for fiber in spec.fibers:
            self.incoming_fibers[fiber.dst].append(fiber.src)

        self.strategies = {area.name: self._make_strategy(area) for area in spec.areas}
        self.activations: dict[str, np.ndarray] = {
            area.name: np.array([], dtype=np.int64) for area in spec.areas
        }
        self.inhibition_state = InhibitionState()
        self.step_count = 0

    def _resolve_step_order(self, step_order: Any) -> list[str]:
        if isinstance(step_order, (list, tuple)):
            order = [str(name) for name in step_order]
            unknown = [name for name in order if name not in self.areas_by_name]
            if unknown:
                raise ValueError(f"step_order contains unknown areas: {unknown}")
            if len(set(order)) != len(order):
                raise ValueError("step_order contains duplicate area names")
            missing = [name for name in self.area_names if name not in order]
            return order + missing
        return list(self.area_names)

    def _make_strategy(self, area: AreaSpec):
        if area.dynamics_type == "feedforward":
            return FeedforwardStrategy(n=area.n)

        if area.dynamics_type == "recurrent":
            recurrent_weights = self.weights.get((area.name, area.name))
            if recurrent_weights is None:
                recurrent_weights = csr_matrix((area.n, area.n), dtype=np.float64)
            return RecurrentStrategy(recurrent_weights=recurrent_weights)

        if area.dynamics_type == "refracted":
            return RefractedStrategy(n=area.n)

        raise ValueError(f"unknown dynamics_type: {area.dynamics_type}")

    def step(
        self,
        external_stimuli: dict[str, np.ndarray] | None = None,
        plasticity_on: bool = True,
    ) -> StepResult:
        self.step_count += 1

        stimuli = external_stimuli or {}
        beta = self.spec.beta

        for area_name in self.step_order:
            if self.inhibition_state.is_inhibited(area_name):
                continue

            area = self.areas_by_name[area_name]
            previous_activations = self.activations[area_name]
            total_input = np.zeros(area.n, dtype=np.float64)

            for src_name in self.incoming_fibers[area_name]:
                key = (src_name, area_name)
                src_activations = self.activations[src_name]
                if src_activations.size == 0:
                    continue
                incoming_sum = self.weights[key][src_activations].sum(axis=0)
                total_input += np.asarray(incoming_sum).ravel()

            total_input += self.strategies[area_name].dynamics_contribution(
                area_name=area_name,
                activations=previous_activations,
                context={"step": self.step_count},
            )

            ext = stimuli.get(area_name)
            if ext is not None:
                total_input += np.asarray(ext, dtype=np.float64)

            new_activations = k_cap(total_input, area.k, self.rng).astype(np.int64, copy=False)

            if plasticity_on:
                for src_name in self.incoming_fibers[area_name]:
                    hebbian_update(
                        self.weights[(src_name, area_name)],
                        pre_firing=self.activations[src_name],
                        post_firing=new_activations,
                        beta=beta,
                    )

                self.strategies[area_name].update_state(
                    area_name=area_name,
                    firing=new_activations,
                    total_input=total_input,
                    plasticity=beta,
                    context={"pre_activations": previous_activations, "step": self.step_count},
                )

            self.activations[area_name] = new_activations

        assemblies = {
            area_name: Assembly(area_name=area_name, indices=self.activations[area_name])
            for area_name in self.area_names
        }
        return StepResult(assemblies=assemblies, step=self.step_count)

    def rollout(
        self,
        stimulus_schedule: list[dict[str, np.ndarray] | None] | None,
        t_internal: int,
        plasticity_on: bool = True,
        trace_spec: Any | None = None,
    ) -> Trace | None:
        if trace_spec is None:
            for t in range(t_internal):
                stimulus = None
                if stimulus_schedule:
                    stimulus = stimulus_schedule[t % len(stimulus_schedule)]
                self.step(external_stimuli=stimulus, plasticity_on=plasticity_on)
            return None

        trace = Trace(steps=[])
        only_areas = getattr(trace_spec, "areas", None)

        for t in range(t_internal):
            stimulus = None
            if stimulus_schedule:
                stimulus = stimulus_schedule[t % len(stimulus_schedule)]

            result = self.step(external_stimuli=stimulus, plasticity_on=plasticity_on)
            if only_areas is None:
                trace.steps.append(result)
                continue

            filtered = {
                name: assembly
                for name, assembly in result.assemblies.items()
                if name in only_areas
            }
            trace.steps.append(StepResult(assemblies=filtered, step=result.step))

        return trace

    def inhibit(self, area_name: str) -> None:
        self.inhibition_state.inhibit(area_name)

    def disinhibit(self, area_name: str) -> None:
        self.inhibition_state.disinhibit(area_name)

    def get_assembly(self, area_name: str) -> Assembly:
        return Assembly(area_name=area_name, indices=self.activations[area_name])

    def normalize(self, area_name: str | None = None) -> None:
        if area_name is None:
            for area in self.spec.areas:
                normalize_weights(self.weights, area.name, self.spec)
        else:
            normalize_weights(self.weights, area_name, self.spec)
