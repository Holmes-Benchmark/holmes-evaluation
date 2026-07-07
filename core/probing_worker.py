import binascii
import glob
import os
import shutil
import uuid
from pathlib import Path

import numpy
import pandas
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from sklearn.dummy import DummyRegressor
from torch.utils.data import Subset

from model.probing_model import LinearProbingModel
from utilities.data_loading import ProbingDataset, get_unique_inputs
from utilities.seed_util import seed_all
import redis
class ProbeWorker:

    def __init__(self, hyperparameter: dict, train_dataset: ProbingDataset, dev_dataset: ProbingDataset, test_dataset: ProbingDataset, n_layers: int, probe_name: str, project_prefix:str, dump_preds:bool, force:bool, result_folder:str, logging:str, cache_folder:str = None):
        self.hyperparameter = hyperparameter
        seed_all(self.hyperparameter["seed"])

        self.dump_preds = dump_preds
        self.result_folder = result_folder
        self.cache_folder = cache_folder
        self.force = force
        self.encoding = self.hyperparameter["encoding"]
        self.probe_name = probe_name
        self.batch_size = hyperparameter["batch_size"]
        self.n_layers = n_layers
        self.project_prefix = project_prefix
        self.precision = 16 if self.encoding != "full" else 32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpus = 1 if torch.cuda.is_available() else 0
        self.is_regression = self.hyperparameter["num_labels"] == 1
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.logging = logging

        self.hyperparameter["gpus"] = self.gpus
        self.hyperparameter["device"] = self.device



    def get_local_run_id(self):
        # NOTE: include fold / generation_id / layer_id so that runs executed in
        # parallel (Pool) never share a CSVLogger directory. Without these, runs
        # differing only by fold/layer/generation collide on Lightning's version_N
        # auto-numbering and the redis-path `rm -rf {log_dir}` cleanup races with
        # another worker's metrics.csv finalize -> intermittent FileNotFoundError.
        run_id = "/".join([
            self.hyperparameter["model_name"].replace('/', "__"),
            self.hyperparameter["encoding"],
            self.hyperparameter["control_task_type"],
            str(self.hyperparameter["sample_size"]),
            str(self.hyperparameter["seed"]),
            str(self.hyperparameter["num_hidden_layers"]),
            str(self.hyperparameter.get("layer_id", "")),
            str(self.hyperparameter.get("generation_id", "")),
            str(self.hyperparameter.get("fold", "")),
        ])

        return run_id

    def get_logger(self):
        if self.logging == "local" and self.project_prefix != "":
            return CSVLogger(save_dir=self.result_folder, name=f"{self.project_prefix}-{self.probe_name}/{self.get_local_run_id()}")
        elif self.logging == "local":
            return CSVLogger(save_dir=self.result_folder, name=f"{self.probe_name}/{self.get_local_run_id()}")
        elif self.logging == "redis" and self.project_prefix != "":
            return  CSVLogger(save_dir=self.cache_folder, name=f"{self.probe_name}/{self.get_local_run_id()}")
        elif self.logging == "redis" and self.project_prefix == "":
            return  CSVLogger(save_dir=self.cache_folder, name=f"{self.project_prefix}-{self.probe_name}/{self.get_local_run_id()}")
        elif self.logging == "wandb" and self.project_prefix != "":
            return WandbLogger(project=self.project_prefix + "-" + self.probe_name, dir=self.cache_folder)

            logger.experiment.config["result_folder"] = self.result_folder
            logger.experiment.config["cache_folder"] = self.cache_folder
        else:
            return WandbLogger(project=self.probe_name, dir=self.cache_folder)
            logger.experiment.config["result_folder"] = self.result_folder
            logger.experiment.config["cache_folder"] = self.cache_folder


    def mark_run_as_done(self, logger):
        if self.logging == "wandb":
            logger.experiment.config["result"] = "done"
            logger.experiment.finish()
        elif self.logging == "local":
            os.system(f"mv {logger.log_dir} {logger.root_dir}/done")

    def get_unique_inputs(self, dataset):
        return dataset.unique_inputs
    def log_params(self, logger, params):
        if self.logging == "wandb":
            for k, v in params.items():
                logger.experiment.config[k] = v

    def log_redis_metrics(self, fields, metrics):
        r = redis.Redis(host=self.hyperparameter["redis_server"], port=self.hyperparameter["redis_port"], db=0)

        path = "/" +  "/".join([
            str(v)
            for k, v in fields.items()
        ]) + "__RUN"

        steps = list(set([
            key.split("step_")[1]
            for key in metrics.keys()
            if "step" in key
        ]))

        for step in steps:
            step_element = {
                key.replace("summary.", "").replace(" ", "_"): value
                for key, value in metrics.items()
                if f"step_{step}" in key
            }
            step_path = f"{path}__STEP__{step}"

            r.hset(step_path, mapping=step_element)

        element = {
            key.replace("summary.", "").replace(" ", "_"): value
            for key, value in metrics.items()
            if "full" in key or "dump_id" in key or "compression" in key or "uniform_length" in key or "minimum_description_length" in key
        }


        r.hset(path, mapping=element)
class GeneralProbeWorker(ProbeWorker):

    def __init__(self, hyperparameter: dict, train_dataset: ProbingDataset, dev_dataset: ProbingDataset, test_dataset: ProbingDataset, n_layers: int, probe_name: str, project_prefix:str, dump_preds:bool, force:bool, result_folder:str, logging:str, cache_folder:str = None):
        super().__init__(hyperparameter, train_dataset, dev_dataset, test_dataset, n_layers, probe_name, project_prefix, dump_preds, force, result_folder, logging, cache_folder)
        self.probing_model = LinearProbingModel


    def train_run(self, log_dir, logger=None):

        batch_size = self.hyperparameter["batch_size"]

        probing_model = self.probing_model(
            hyperparameter=self.hyperparameter, unique_inputs=self.train_dataset.unique_inputs
        ).to(self.device)

        train_dataloader = probing_model.get_dataloader(self.train_dataset, batch_size, shuffle=True)
        dev_dataloader = probing_model.get_dataloader(self.dev_dataset, 300, shuffle=False)
        test_dataloader = probing_model.get_test_dataloader(self.test_dataset, 300, shuffle=False)

        probing_model.hyperparameter["training_steps"] = self.hyperparameter["training_steps"] = len(train_dataloader) * 20
        probing_model.hyperparameter["warmup_steps"] = self.hyperparameter["warmup_steps"] = self.hyperparameter["training_steps"] * self.hyperparameter["warmup_rate"]

        trainer = Trainer(
            logger=logger, max_epochs=20, accelerator="auto", devices=1, precision=self.precision,
            num_sanity_val_steps=0, deterministic=False,
            callbacks=[ModelCheckpoint(monitor="val loss",  mode="min", dirpath=log_dir), EarlyStopping(monitor="val loss",  mode="min", patience=10)]
        )

        trainer.fit(model=probing_model, train_dataloaders=[train_dataloader], val_dataloaders=[dev_dataloader])

        trainer.test(ckpt_path="best", dataloaders=[test_dataloader])

        print("pred done")

        # For distribution probes the dataset label is a full probability vector; store the
        # model's compact top-1 target instead so preds.csv stays small.
        label_source = (probing_model.test_labels
                        if getattr(probing_model, "is_distribution", False)
                        else self.test_dataset.labels)
        test_predictions = [
            (instance_input, pred, instance_label, loss)
            for instance_input, instance_label, pred, loss in zip(self.test_dataset.inputs, label_source, probing_model.test_preds, probing_model.test_losses)
        ]

        test_prediction_frame = pandas.DataFrame(test_predictions)
        test_prediction_frame.columns = ["instance", "pred", "label", "loss"]

        return test_prediction_frame, probing_model


    def run_fold(self):

        logger = self.get_logger()

        if self.logging == "local":
            log_dir = logger.log_dir
            result_log_dir = log_dir

            if os.path.exists(f"{logger.root_dir}/done") and not self.force:
                print(f"Already done at {logger.root_dir}/done")
                return "Done"
        elif self.logging == "redis":
            log_dir = logger.log_dir
            random_id = str(uuid.uuid4())
            result_log_dir = f"{self.result_folder}/{random_id}"
            os.system(f"mkdir -p {result_log_dir}")
        else:
            #if check_wandb_run(self.hyperparameter, logger.experiment.project) and not self.force:
            #    print(f"Already done.")
            #    return "Done"

            log_dir = f"{self.result_folder}/{logger.experiment.id}"
            result_log_dir = log_dir.copy()

        os.system("mkdir -p " + log_dir)

        self.hyperparameter["dump_id"] = log_dir
        self.hyperparameter["cache_folder"] = self.cache_folder
        self.hyperparameter["result_folder"] = self.result_folder

        prediction_frame, probing_model = self.train_run(log_dir=result_log_dir, logger=logger)

        if self.dump_preds:
            prediction_frame.to_csv(result_log_dir +"/preds.csv")

        if self.logging == "redis":
            metrics = probing_model.best_test_metrics
            metrics["dump_id"] = result_log_dir
            self.log_redis_metrics(self.hyperparameter["redis_run_fields"], metrics)
            try:
                logger.finalize("success")
            except Exception:
                pass
            shutil.rmtree(log_dir, ignore_errors=True)
        self.mark_run_as_done(logger=logger)

        return "Done"



class MDLProbeWorker(GeneralProbeWorker):

    def __init__(self, hyperparameter: dict, train_dataset: ProbingDataset, dev_dataset: ProbingDataset, test_dataset: ProbingDataset, n_layers: int, probe_name: str, project_prefix:str, dump_preds:bool, force:bool, result_folder:str, logging:str, cache_folder:str = None):
        super().__init__(hyperparameter, train_dataset, dev_dataset, test_dataset, n_layers, probe_name, project_prefix, dump_preds, force, result_folder, logging, cache_folder)



    def train_mdl_run(self, train_dataset, dev_online_dataset, dev_dataset, test_dataset, log_dir, logger=None):



        batch_size = self.hyperparameter["batch_size"]

        probing_model = self.probing_model(
            hyperparameter=self.hyperparameter,
        ).to(self.device)

        train_dataloader = probing_model.get_dataloader(train_dataset, batch_size, shuffle=True)
        dev_online_dataloader = probing_model.get_dataloader(dev_online_dataset, 300, shuffle=False)
        dev_dataloader = probing_model.get_dataloader(dev_dataset, 300, shuffle=False)
        test_dataloader = probing_model.get_test_dataloader(test_dataset, 300,shuffle=False)

        probing_model.hyperparameter["training_steps"] = self.hyperparameter["training_steps"] = len(train_dataloader) * 20
        probing_model.hyperparameter["warmup_steps"] = self.hyperparameter["warmup_steps"] = self.hyperparameter["training_steps"] * self.hyperparameter["warmup_rate"]

        trainer = Trainer(
            logger=logger, max_epochs=20, accelerator="auto", devices=1, precision=self.precision,
            num_sanity_val_steps=0, deterministic=False, gradient_clip_val=1.0,
            callbacks=[ModelCheckpoint(monitor="val_ref",  mode="max", dirpath=log_dir), EarlyStopping(monitor="val_ref",  mode="max", patience=4)]
        )

        trainer.fit(model=probing_model, train_dataloaders=[train_dataloader], val_dataloaders=[dev_dataloader])

        dev_metrics = trainer.validate(ckpt_path="best", dataloaders=[dev_online_dataloader])

        test_metrics = trainer.test(ckpt_path="best", dataloaders=[test_dataloader])[0]


        summed_loss = dev_metrics[0]["val loss sum"]

        return summed_loss, test_metrics



    def run_linear_task_fraction(self, fraction:int, ref_dataset, dev_dataset, test_dataset, log_dir:str=None):

        fraction_length = int(len(ref_dataset) * fraction)

        if fraction_length == 0:
            return 0, [], [], [], [], {}, 0

        train_dataset = Subset(ref_dataset, list(range(0, fraction_length)))
        dev_online_dataset = Subset(ref_dataset, list(range(fraction_length, fraction_length*2)))

        print(len(ref_dataset))
        print(fraction_length)
        print(fraction_length)


        summed_loss, test_metrics = self.train_mdl_run(
            train_dataset, dev_online_dataset,
            dev_dataset, test_dataset, log_dir + "/frac-" + str(fraction_length), logger=False
        )

        return summed_loss, test_metrics, len(dev_online_dataset)


    def run_mdl_tasks(self, fractions, log_dir, ref_dataset, dev_dataset, test_dataset):

        fraction_losses = []
        collected_test_metrics = []
        fraction_lengths = []
        overall_test_metrics = []


        for fraction in fractions:


            summed_loss, test_metrics, fraction_length = self.run_linear_task_fraction(
                fraction=fraction, ref_dataset=ref_dataset, dev_dataset=dev_dataset,
                test_dataset=test_dataset, log_dir=log_dir
            )

            if fraction_length == 0:
                continue


            if self.is_regression:
                collected_test_metrics.append({
                    "pearson": test_metrics.get("full test pearson", 0),
                })
            else:
                collected_test_metrics.append({
                    "acc": test_metrics.get("full test acc", 0),
                    "f1": test_metrics.get("full test f1", 0),
                })

            test_metrics["fraction"] = fraction

            overall_test_metrics.append(test_metrics)

            fraction_lengths.append(fraction_length)
            fraction_losses.append(summed_loss)



        os.system("rm -rf " + log_dir + "/frac*")

        first_portion_size = min([ele for ele in fraction_lengths if ele > 0])

        if self.is_regression:
            labels = [ele for ele in ref_dataset.labels]

            dummy_model = DummyRegressor(strategy="mean")
            dummy_model.fit(labels, labels)
            samples_labels = dummy_model.predict(labels)

            uniform_code_length = float(torch.nn.SmoothL1Loss(reduction="sum")(torch.tensor(samples_labels), torch.tensor(labels)))

            minimum_description_length = first_portion_size * (uniform_code_length / len(ref_dataset)) + sum(fraction_losses)
            compression = uniform_code_length/minimum_description_length

        else:
            uniform_code_length = len(ref_dataset) * numpy.log2(self.hyperparameter["num_labels"])
            minimum_description_length = first_portion_size * numpy.log2(self.hyperparameter["num_labels"]) + sum(fraction_losses)
            compression = uniform_code_length/minimum_description_length

        return uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics


    def train_run(self, log_dir, logger):

        batch_size = self.hyperparameter["batch_size"]

        unique_inputs = self.train_dataset.unique_inputs

        probing_model = self.probing_model(
            hyperparameter=self.hyperparameter, unique_inputs=unique_inputs
        ).to(self.device)

        train_dataloader = probing_model.get_dataloader(self.train_dataset, batch_size, shuffle=True)
        dev_dataloader = probing_model.get_dataloader(self.dev_dataset, 300, shuffle=False)
        test_dataloader = probing_model.get_test_dataloader(self.test_dataset, 300, shuffle=False)

        probing_model.hyperparameter["training_steps"] = self.hyperparameter["training_steps"] = len(train_dataloader) * 20
        probing_model.hyperparameter["warmup_steps"] = self.hyperparameter["warmup_steps"] = self.hyperparameter["training_steps"] * self.hyperparameter["warmup_rate"]

        trainer = Trainer(
            logger=logger, max_epochs=20, accelerator="auto", devices=1, precision=self.precision,
            num_sanity_val_steps=0, deterministic=False, gradient_clip_val=1.0,
            callbacks=[ModelCheckpoint(monitor="val_ref",  mode="max", dirpath=log_dir), EarlyStopping(monitor="val_ref",  mode="max", patience=5)]
        )

        trainer.fit(model=probing_model, train_dataloaders=[train_dataloader], val_dataloaders=[dev_dataloader])

        trainer.test(ckpt_path="best", dataloaders=[test_dataloader])

        uniform_code_length, minimum_description_length, compression, fraction_losses, fraction_lengths, collected_test_metrics = self.run_mdl_tasks(
            fractions=[1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2],
            log_dir=log_dir,
            ref_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
        )

        self.save_mdl_metrics(
            logger, uniform_code_length, minimum_description_length, compression,
            fraction_losses, fraction_lengths, collected_test_metrics
        )
        print("pred done")

        # For distribution probes the dataset label is a full probability vector; store the
        # model's compact top-1 target instead so preds.csv stays small.
        label_source = (probing_model.test_labels
                        if getattr(probing_model, "is_distribution", False)
                        else self.test_dataset.labels)
        test_predictions = [
            (instance_input, pred, instance_label, loss)
            for instance_input, instance_label, pred, loss in zip(self.test_dataset.inputs, label_source, probing_model.test_preds, probing_model.test_losses)
        ]

        test_prediction_frame = pandas.DataFrame(test_predictions)
        test_prediction_frame.columns = ["instance", "pred", "label", "loss"]

        return test_prediction_frame, probing_model


    def save_mdl_metrics(
            self, logger, uniform_code_length, minimum_description_length, compression,
            fraction_losses, fraction_lengths, collected_test_metrics
    ):
        metrics = {
            "uniform_length": uniform_code_length,
            "minimum_description_length": minimum_description_length,
            "compression": compression,
        }

        for i, (fraction_loss, fraction_length, test_metrics) in enumerate(zip(fraction_losses, fraction_lengths, collected_test_metrics)):
            if i > 0:
                for metric in test_metrics.keys():
                    metrics["z_test_" + str(i) + "_" + metric + "_step_" +str(fraction_length)] = test_metrics[metric]
                    metrics["z_loss_" + str(i) + "_" + metric + "_step_" +str(fraction_length)] = fraction_loss



        if self.logging == "redis":
            self.log_redis_metrics(self.hyperparameter["redis_run_fields"], metrics)
        else:
            logger.log_metrics(metrics)

