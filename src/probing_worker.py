import glob
import os
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
from utils.data_loading import ProbingDataset, get_unique_inputs
from utils.experiment_util import check_wandb_run
from utils.seed_util import seed_all

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
        run_id = "/".join([
            self.hyperparameter["model_name"].replace('/', "__"),
            self.hyperparameter["encoding"],
            self.hyperparameter["control_task_type"],
            str(self.hyperparameter["sample_size"]),
            str(self.hyperparameter["seed"]),
            str(self.hyperparameter["num_hidden_layers"]),
        ])

        return run_id

    def get_logger(self):
        if self.logging == "local" and self.project_prefix != "":
            return CSVLogger(save_dir=self.result_folder, name=f"{self.project_prefix}-{self.probe_name}/{self.get_local_run_id()}")
        elif self.logging == "local":
            return CSVLogger(save_dir=self.result_folder, name=f"{self.probe_name}/{self.get_local_run_id()}")
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

        test_predictions = [
            (instance_input, pred, instance_label, loss, "seen" if seen_index else "unseen")
            for instance_input, instance_label, pred, loss, seen_index in zip(self.test_dataset.inputs, self.test_dataset.labels, probing_model.test_preds, probing_model.test_losses, probing_model.test_seen_indices)
        ]

        test_prediction_frame = pandas.DataFrame(test_predictions)
        test_prediction_frame.columns = ["instance", "pred", "label", "loss", "seen"]

        return test_prediction_frame, probing_model


    def run_fold(self):

        logger = self.get_logger()

        if self.logging == "local":
            log_dir = logger.log_dir
            if os.path.exists(f"{logger.root_dir}/done") and not self.force:
                print(f"Already done at {logger.root_dir}/done")
                return "Done"
        else:
            #if check_wandb_run(self.hyperparameter, logger.experiment.project) and not self.force:
            #    print(f"Already done.")
            #    return "Done"

            log_dir = f"{self.result_folder}/{logger.experiment.id}"

        os.system("mkdir -p " + log_dir)

        self.hyperparameter["dump_id"] = log_dir
        self.hyperparameter["cache_folder"] = self.cache_folder
        self.hyperparameter["result_folder"] = self.result_folder

        prediction_frame, probing_model = self.train_run(log_dir=log_dir, logger=logger)

        if self.dump_preds:
            prediction_frame.to_csv(log_dir +"/preds.csv")

        self.mark_run_as_done(logger=logger)

        return "Done"



class MDLProbeWorker(GeneralProbeWorker):

    def __init__(self, hyperparameter: dict, train_dataset: ProbingDataset, dev_dataset: ProbingDataset, test_dataset: ProbingDataset, n_layers: int, probe_name: str, project_prefix:str, dump_preds:bool, force:bool, result_folder:str, logging:str, cache_folder:str = None):
        super().__init__(hyperparameter, train_dataset, dev_dataset, test_dataset, n_layers, probe_name, project_prefix, dump_preds, force, result_folder, logging, cache_folder)



    def train_mdl_run(self, train_dataset, dev_online_dataset, dev_online_seen_indices, dev_online_unseen_indices, dev_dataset, test_dataset, log_dir, logger=None):

        dev_online_seen_indices = [ele for ele in dev_online_seen_indices if ele < len(dev_online_dataset)]
        dev_online_unseen_indices = [ele for ele in dev_online_unseen_indices if ele < len(dev_online_dataset)]


        batch_size = self.hyperparameter["batch_size"]

        test_seen_indices = test_dataset.get_seen_indices()
        test_unseen_indices = test_dataset.get_unseen_indices()

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

        if len(dev_online_seen_indices) > 0:
            dev_seen_losses = probing_model.dev_losses[dev_online_seen_indices]
        else:
            dev_seen_losses = []

        if len(dev_online_unseen_indices) > 0:
            dev_unseen_losses = probing_model.dev_losses[dev_online_unseen_indices]
        else:
            dev_unseen_losses = []

        test_metrics = trainer.test(ckpt_path="best", dataloaders=[test_dataloader])[0]

        if len(test_seen_indices) > 0:
            test_seen_preds = probing_model.test_raw_preds[test_seen_indices]
            test_seen_labels = probing_model.test_labels[test_seen_indices]
        else:
            test_seen_preds = test_seen_labels = []

        if len(test_unseen_indices) > 0:
            test_unseen_preds = probing_model.test_raw_preds[test_unseen_indices]
            test_unseen_labels = probing_model.test_labels[test_unseen_indices]
        else:
            test_unseen_preds = test_unseen_labels = []

        for name, func in probing_model.metrics.items():
            if len(test_seen_preds):
                test_metrics["seen " + name] = float(func(test_seen_preds.argmax(dim=1), test_seen_labels))
            else:
                test_metrics["seen " + name] = -1

            if len(test_unseen_preds):
                test_metrics["unseen " + name] = float(func(test_unseen_preds.argmax(dim=1), test_unseen_labels))
            else:
                test_metrics["unseen " + name] = -1


        summed_loss = dev_metrics[0]["val loss sum"]

        return summed_loss, dev_seen_losses, dev_unseen_losses, dev_online_seen_indices, dev_online_unseen_indices, test_metrics



    def run_linear_task_fraction(self, fraction:int, ref_dataset, dev_dataset, test_dataset, log_dir:str=None):

        fraction_length = int(len(ref_dataset) * fraction)

        if fraction_length == 0:
            return 0, [], [], [], [], {}, 0

        train_dataset = Subset(ref_dataset, list(range(0, fraction_length)))
        dev_online_dataset = Subset(ref_dataset, list(range(fraction_length, fraction_length*2)))

        print(len(ref_dataset))
        print(fraction_length)
        print(fraction_length)

        train_inputs = ref_dataset.inputs[:fraction_length]
        train_unique_inputs = get_unique_inputs(train_inputs)
        dev_online_inputs = ref_dataset.inputs[fraction_length:fraction_length*2 - 1]
        dev_online_seen_indices = [i for i, element in enumerate(dev_online_inputs) if tuple([ele[0].lower() for ele in element]) in train_unique_inputs]
        dev_online_unseen_indices = [i for i, element in enumerate(dev_online_inputs) if tuple([ele[0].lower() for ele in element]) not in train_unique_inputs]

        summed_loss, dev_seen_losses, dev_unseen_losses, dev_online_seen_indices, dev_online_unseen_indices, test_metrics = self.train_mdl_run(
            train_dataset, dev_online_dataset, dev_online_seen_indices, dev_online_unseen_indices,
            dev_dataset, test_dataset, log_dir + "/frac-" + str(fraction_length), logger=False
        )

        return summed_loss, dev_seen_losses, dev_unseen_losses, dev_online_seen_indices, dev_online_unseen_indices, test_metrics, len(dev_online_dataset)


    def run_mdl_tasks(self, fractions, log_dir, ref_dataset, dev_dataset, test_dataset):

        fraction_losses = []
        seen_fraction_losses = []
        unseen_fraction_losses = []
        collected_test_metrics = []
        fraction_lengths = []
        seen_fraction_lengths = []
        unseen_fraction_lengths = []
        overall_test_metrics = []

        all_dev_online_seen_indices = []
        all_dev_online_unseen_indices = []

        for fraction in fractions:


            summed_loss, dev_seen_losses, dev_unseen_losses, dev_online_seen_indices, dev_online_unseen_indices, test_metrics, fraction_length = self.run_linear_task_fraction(
                fraction=fraction, ref_dataset=ref_dataset, dev_dataset=dev_dataset,
                test_dataset=test_dataset, log_dir=log_dir
            )

            if fraction_length == 0:
                continue

            all_dev_online_seen_indices.append(dev_online_seen_indices)
            all_dev_online_unseen_indices.append(dev_online_unseen_indices)


            if self.is_regression:
                collected_test_metrics.append({
                    "pearson": test_metrics.get("full test pearson", 0),
                    "seen_pearson": test_metrics.get("seen test pearson", 0),
                    "unseen_pearson": test_metrics.get("unseen test pearson", 0),
                })
            else:
                collected_test_metrics.append({
                    "acc": test_metrics.get("full test acc", 0),
                    "f1": test_metrics.get("full test f1", 0),
                    "seen_acc": test_metrics.get("seen acc", 0),
                    "seen_f1": test_metrics.get("seen f1", 0),
                    "unseen_acc": test_metrics.get("unseen acc", 0),
                    "unseen_f1": test_metrics.get("unseen f1", 0),
                })

            test_metrics["fraction"] = fraction

            overall_test_metrics.append(test_metrics)

            fraction_lengths.append(fraction_length)
            fraction_losses.append(summed_loss)

            if len(dev_seen_losses) == 0:
                seen_fraction_lengths.append(0)
                seen_fraction_losses.append(0)
            else:
                seen_fraction_losses.append(dev_seen_losses.sum())
                seen_fraction_lengths.append(len(dev_seen_losses))

            if len(dev_unseen_losses) == 0:
                unseen_fraction_lengths.append(0)
                unseen_fraction_losses.append(0)
            else:
                unseen_fraction_lengths.append(len(dev_unseen_losses))
                unseen_fraction_losses.append(dev_unseen_losses.sum())


        os.system("rm -rf " + log_dir + "/frac*")

        first_portion_size = min([ele for ele in fraction_lengths if ele > 0])

        if self.is_regression:
            labels = [ele for ele in ref_dataset.labels]

            dummy_model = DummyRegressor(strategy="mean")
            dummy_model.fit(labels, labels)
            samples_labels = dummy_model.predict(labels)

            uniform_code_length = float(torch.nn.MSELoss(reduction="sum")(torch.tensor(samples_labels), torch.tensor(labels)))

            minimum_description_length = first_portion_size * (uniform_code_length / len(ref_dataset)) + sum(fraction_losses)
            compression = uniform_code_length/minimum_description_length
            seen_compression = 0
            unseen_compression = 0

        else:
            uniform_code_length = len(ref_dataset) * numpy.log2(self.hyperparameter["num_labels"])
            minimum_description_length = first_portion_size * numpy.log2(self.hyperparameter["num_labels"]) + sum(fraction_losses)
            seen_minimum_description_length = first_portion_size * numpy.log2(self.hyperparameter["num_labels"]) + sum(seen_fraction_losses)
            unseen_minimum_description_length = first_portion_size * numpy.log2(self.hyperparameter["num_labels"]) + sum(unseen_fraction_losses)

            seen_uniform_code_length = sum(seen_fraction_lengths) * numpy.log2(self.hyperparameter["num_labels"])
            unseen_uniform_code_length = sum(unseen_fraction_lengths) * numpy.log2(self.hyperparameter["num_labels"])
            compression = uniform_code_length/minimum_description_length
            seen_compression = seen_uniform_code_length/seen_minimum_description_length
            unseen_compression = unseen_uniform_code_length/unseen_minimum_description_length

        return uniform_code_length, minimum_description_length, compression, seen_compression, unseen_compression, fraction_losses, fraction_lengths, collected_test_metrics


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

        uniform_code_length, minimum_description_length, compression, seen_compression, unseen_compression, fraction_losses, fraction_lengths, collected_test_metrics = self.run_mdl_tasks(
            fractions=[1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2],
            log_dir=log_dir,
            ref_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
        )

        self.save_mdl_metrics(
            logger, uniform_code_length, minimum_description_length, compression, seen_compression, unseen_compression,
            fraction_losses, fraction_lengths, collected_test_metrics
        )
        print("pred done")

        test_predictions = [
            (instance_input, pred, instance_label, loss, "seen" if seen_index else "unseen")
            for instance_input, instance_label, pred, loss, seen_index in zip(self.test_dataset.inputs, self.test_dataset.labels, probing_model.test_preds, probing_model.test_losses, probing_model.test_seen_indices)
        ]

        test_prediction_frame = pandas.DataFrame(test_predictions)
        test_prediction_frame.columns = ["instance", "pred", "label", "loss", "seen"]

        return test_prediction_frame, probing_model


    def save_mdl_metrics(
            self, logger, uniform_code_length, minimum_description_length, compression, seen_compression, unseen_compression,
            fraction_losses, fraction_lengths, collected_test_metrics
    ):
        metrics = {
            "uniform_length": uniform_code_length,
            "minimum_description_length": minimum_description_length,
            "compression": compression,
            "seen_compression": seen_compression,
            "unseen_compression": unseen_compression,
        }


        for i, (fraction_loss, fraction_length, test_metrics) in enumerate(zip(fraction_losses, fraction_lengths, collected_test_metrics)):
            if i > 0:
                for metric in test_metrics.keys():
                    metrics["z_test_" + str(i) + "_" + metric + "_step_" +str(fraction_length)] = test_metrics[metric]
                    metrics["z_loss_" + str(i) + "_" + metric + "_step_" +str(fraction_length)] = fraction_loss

        logger.log_metrics(metrics)

