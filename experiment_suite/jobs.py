from __future__ import annotations

from dataclasses import dataclass

from experiment_suite.config import ExperimentCondition, ModelConfig, SuiteConfig


@dataclass(frozen=True)
class ExperimentJob:
    suite_name: str
    output_dir: str
    family: str
    model: ModelConfig
    seed: int
    condition: ExperimentCondition


def expand_jobs(config: SuiteConfig) -> list[ExperimentJob]:
    jobs: list[ExperimentJob] = []
    for family, model_configs in config.models.items():
        for model_config in model_configs:
            for seed in config.seeds:
                for condition in config.conditions:
                    jobs.append(
                        ExperimentJob(
                            suite_name=config.suite_name,
                            output_dir=config.output_dir,
                            family=family,
                            model=model_config,
                            seed=seed,
                            condition=condition,
                        )
                    )
    return jobs
