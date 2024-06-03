import os
import traceback
import uuid
from typing import Dict

import click
import torch
import yaml
from itertools import product

from defs.control_task_types import CONTROL_TASK_TYPES
from utils.config_loading import load_base_config, get_load_configs
from utils.data_loading import load_probe_file, load_data, load_datasets
from utils.session_utils import setup_session, clean_session, run_probe_with_params


def get_hyperparameters(hyperparameters:Dict):
    params = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]
    params = [
        param
        for param in params
        if not (param["num_hidden_layers"] == 0 and param["hidden_dim"] > 0) and not (param["num_hidden_layers"] > 0 and param["hidden_dim"] == 0)
    ]
    return params



@click.command()
@click.option('--config_file_path', type=str, default='../holmes-datasets/zorro-quantifiers-superlative/config-none.yaml')
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--model_precision', type=str, default="full")
@click.option('--seeds', type=str, default="0,1,2,3,4")
@click.option('--num_hidden_layers', type=str, default="0")
@click.option('--batch_size', type=int, default=16)
@click.option('--run_probe', type=bool, default=True)
@click.option('--run_mdl_probe', type=bool, default=True)
@click.option('--project_prefix', type=str, default="")
@click.option('--dump_preds', is_flag=True, default=False)
@click.option('--force', is_flag=True, default=False)
@click.option('--dump_folder', type=str, default="../dumps")
@click.option('--result_folder', type=str, default="../results")
def main(
        config_file_path, model_name, encoding, seeds, num_hidden_layers,
        batch_size, run_probe, run_mdl_probe, project_prefix, dump_preds, force,
        dump_folder, result_folder
):
    base_path = "/".join(config_file_path.split("/")[:-1]) + "/samples.csv"

    file_stream = open(config_file_path, "r")
    config = yaml.safe_load(file_stream)

    control_task_type = CONTROL_TASK_TYPES[config["control_task_type"]]

    base_config = load_base_config(
        config=config, encoding=encoding,
        seeds=seeds, num_hidden_layers=num_hidden_layers,
        model_name=model_name, batch_size=batch_size,
        control_task_type=control_task_type, project_prefix=project_prefix
    )

    configs = get_load_configs(
        base_config, run_probe, run_mdl_probe
    )

    session_id = "session-" + str(uuid.uuid4())
    setup_session(session_id)


    torch.set_num_threads(1)

    default_params = []

    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    probe_frame = load_probe_file(base_path, control_task_type)

    dump_id = "__".join([model_name, encoding, configs[0]["control_task_type"].name, configs[0]["probe_name"], str(probe_frame.shape[0]), "False"])

    probing_frames = load_data(dump_folder, dump_id, control_task_type, scalar_mixin=False)

    input_dim = probing_frames[0]["train"].iloc[0]["inputs_encoded"].shape[-1]
    for fold in range(len(probing_frames)):
        train_dataset, dev_dataset, test_dataset = load_datasets(probing_frames[fold])

        for config in configs:
            config["sample_size"] = probe_frame.shape[0]
            hyperparameters = get_hyperparameters(config["hyperparameters"])

            for hyperparameter in hyperparameters:
                hyperparameter["fold"] = fold
                param_ele = {
                    "hyperparameter": hyperparameter,
                    "input_dim": input_dim,
                    "n_layers": 1,
                    "result_folder": result_folder,
                    **config
                }

                default_params.append((param_ele, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix))


    for param in default_params:
        run_probe_with_params(*param)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        clean_session()
    finally:
        clean_session()