import copy
from typing import Dict

from torch.optim import Adam

from defs.control_task_types import CONTROL_TASK_TYPES
from defs.default_config import DEFAULT_CONFIG
from defs.probe_task_types import PROBE_TASK_TYPES

optimizers = {
    "Adam": Adam
}

def get_load_configs(
        base_config, run_probe, run_mdl_probe
):

    configs = []

    if run_probe and not run_mdl_probe:
        tmp_config = copy.deepcopy(base_config)
        tmp_config["probe_type"] = "linear"
        configs.append(tmp_config)

    if run_probe and run_mdl_probe or not run_probe and run_mdl_probe:
        tmp_config = copy.deepcopy(base_config)
        tmp_config["probe_type"] = "linear_mdl"
        configs.append(tmp_config)

    return configs


def load_base_config(
        config, seeds, num_hidden_layers,
        model_name, batch_size, control_task_type, project_prefix, encoding
) -> Dict:

    if "probe_task_type" in config:
        config["probe_task_type"] = PROBE_TASK_TYPES(config["probe_task_type"])

    if "control_task_type" in config:
        config["control_task_type"] = CONTROL_TASK_TYPES(config["control_task_type"])

    if "optimizer" in config:
        config["optimizer"] = optimizers[config["optimizer"]]

    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    config["encoding"] = encoding

    seeds = [int(seed) for seed in seeds.split(",")]
    num_hidden_layers = [int(ele) for ele in num_hidden_layers.split(",")]

    config["hyperparameters"]["seed"] = seeds

    config["model_name"] = model_name
    config["project_prefix"] = project_prefix

    if not num_hidden_layers is None:
        config["hyperparameters"]["num_hidden_layers"] = num_hidden_layers

    if not batch_size is None:
        config["hyperparameters"]["batch_size"] = [batch_size]

    if control_task_type:
        config["control_task_type"] = CONTROL_TASK_TYPES(control_task_type)

    return config