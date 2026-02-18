from dataclasses import dataclass, asdict
import json
from typing import Union


@dataclass
class Artifact:
    params: dict
    seed: int
    entropy: Union[int, tuple]
    metrics: dict
    pyac_version: str
    timestamp: str


def save_artifacts(artifacts: list[Artifact], path: str) -> None:
    artifacts_as_dicts = [asdict(artifact) for artifact in artifacts]
    with open(path, "w") as f:
        json.dump(artifacts_as_dicts, f, indent=2)


def load_artifacts(path: str) -> list[Artifact]:
    with open(path, "r") as f:
        data = json.load(f)
    
    artifacts = []
    for item in data:
        if isinstance(item["entropy"], list):
            item["entropy"] = tuple(item["entropy"])
        artifacts.append(Artifact(**item))
    
    return artifacts
