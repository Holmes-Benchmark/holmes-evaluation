import os.path
import traceback

import click
import torch
import yaml

from defs.control_task_types import CONTROL_TASK_TYPES
from defs.probe_task_types import PROBE_TASK_TYPES
from utils.data_loading import load_probe_file, dump_data
from utils.session_utils import clean_session


@click.command()
@click.option('--config_file_path', type=str, default='../holmes-datasets/cwi/config-none.yaml')
@click.option('--encoding_batch_size', type=int, default=10)
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--model_precision', type=str, default="full")
@click.option('--dump_folder', type=str, default="../dumps")
@click.option('--force', is_flag=True, default=False)
def main(
        config_file_path, encoding_batch_size, model_name, model_precision, dump_folder, force
):
    base_path = "/".join(config_file_path.split("/")[:-1]) + "/samples.csv"

    file_stream = open(config_file_path, "r")
    config = yaml.safe_load(file_stream)

    control_task_type = CONTROL_TASK_TYPES[config["control_task_type"]]

    probe_frame = load_probe_file(base_path, control_task_type)

    control_task_type = config["control_task_type"]
    probe_task_type = PROBE_TASK_TYPES(config["probe_task_type"])

    dump_id = "__".join([model_name, model_precision, control_task_type, config["probe_name"], str(probe_frame.shape[0]), "False"])
    dump_id = dump_id.replace('/', "__")

    dump_path = f"{dump_folder}/{dump_id}.pickle"

    if os.path.exists(dump_path) and not force:
        print(f"Already encoded at {dump_path}")
        return


    dump_data(probe_frame, probe_task_type, control_task_type, encoding_batch_size, dump_path,
                               model_name, model_precision, scalar_mixin=False)


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
