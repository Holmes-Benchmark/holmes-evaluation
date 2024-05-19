import os

import ray

from probing_worker import GeneralProbeWorker, MDLProbeWorker


def setup_session(session_id):
    session_path = os.environ.get("CACHE_PATH", os.getcwd()) + "/" + session_id


def clean_session():
    if "RAY_SESSION_DIR" in os.environ:
        print("clean", os.environ["RAY_SESSION_DIR"])
        os.system("rm -rf " + os.environ["RAY_SESSION_DIR"])



@ray.remote(num_gpus=1/24)
def ray_run_probe_with_params(params, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix):
    run_probe_with_params(params, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix)

def run_probe_with_params(params, train_dataset, dev_dataset, test_dataset, dump_preds, force, project_prefix):
    hyperparameter = params["hyperparameter"]

    hyperparameter["control_task_type"] = params["control_task_type"].name
    hyperparameter["probe_task_type"] = params["probe_task_type"].name
    hyperparameter["num_labels"] = params["num_labels"]
    hyperparameter["model_name"] = params["model_name"]
    hyperparameter["probe_type"] = params["probe_type"]
    hyperparameter["sample_size"] = params["sample_size"]

    hyperparameter["input_dim"] = params["input_dim"]

    probe_name = params["probes_samples_path"].split("/")[-2]

    if params["probe_type"] == "linear":
        WORKER_CLASS = GeneralProbeWorker
    elif params["probe_type"] == "linear_mdl":
        WORKER_CLASS = MDLProbeWorker

    hyperparameter["encoding"] = params["encoding"]

    worker = WORKER_CLASS(
        train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
        hyperparameter=hyperparameter, project_prefix=project_prefix, n_layers=params["n_layers"],
        probe_name=probe_name, dump_preds=dump_preds, force=force, result_folder=params["result_folder"]
    )

    worker.run_fold()

