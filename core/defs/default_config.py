from torch.optim import Adam

from defs.probe_task_types import PROBE_TASK_TYPES

DEFAULT_CONFIG = {
    "probes_samples_path": "",
    "probe_task_type": PROBE_TASK_TYPES.SENTENCE,
    "num_probe_folds": 4,
    "num_labels": 2,
    "model_name": "bert-base-uncased",
    "output_dim": 768,
    "hyperparameters": {
        "learning_rate": [0.001],
        "batch_size": [64],
        "optimizer": [Adam],
        "hidden_dim": [0, 1000],
        "dropout": [0.2],
        "warmup_rate": [0.1],
        "num_hidden_layers": [0, 1, 2]
    }
}

ALL_HYPERPARAMETERS = list(DEFAULT_CONFIG["hyperparameters"])