from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class ProbingEntry:
    inputs: List[str]
    inputs_encoded: List[torch.tensor]
    context: str
    label: Union[int, float]
    topic: str
    id: int


@dataclass
class ScalarProbingEntry:
    inputs: List[str]
    inputs_encoded: List[List[torch.tensor]]
    context: str
    label: Union[int, float]
    topic: str
    id: int


@dataclass
class ProbingDataset:
    train_entries: List[ProbingEntry]
    dev_entries: List[ProbingEntry]
    test_entries: List[ProbingEntry]


@dataclass
class ScalarProbingDataset:
    train_entries: List[ScalarProbingEntry]
    dev_entries: List[ScalarProbingEntry]
    test_entries: List[ScalarProbingEntry]

@dataclass
class ProbingTask:
    dataset: ProbingDataset

@dataclass
class ScalarProbingTask:
    dataset: ScalarProbingDataset