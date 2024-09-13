import os

import ray

from probing_worker import GeneralProbeWorker, MDLProbeWorker


def setup_session(session_id):
    session_path = os.environ.get("CACHE_PATH", os.getcwd()) + "/" + session_id


def clean_session():
    if "RAY_SESSION_DIR" in os.environ:
        print("clean", os.environ["RAY_SESSION_DIR"])
        os.system("rm -rf " + os.environ["RAY_SESSION_DIR"])



@ray.remote(num_gpus=1/24, num_cpus=3)
def ray_run_probe_with_params(params, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix,  logging="local", probe_name=None):
    run_probe_with_params(params, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix, logging, probe_name)

def run_probe_with_params(params, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix, logging="local", probe_name=None):
    hyperparameter = params["hyperparameter"]

    hyperparameter["input_dim"] = params["input_dim"]

    if probe_name is None:
        probe_name = params["probes_samples_path"].split("/")[-2]

    if hyperparameter["probe_type"] == "linear":
        WORKER_CLASS = GeneralProbeWorker
    elif hyperparameter["probe_type"] == "linear_mdl":
        WORKER_CLASS = MDLProbeWorker


    worker = WORKER_CLASS(
        train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
        hyperparameter=hyperparameter, project_prefix=project_prefix, n_layers=params["n_layers"],
        probe_name=probe_name, dump_preds=dump_preds, force=force, result_folder=params["result_folder"],
        logging=logging, cache_folder=params["cache_folder"]
    )

    worker.run_fold()
