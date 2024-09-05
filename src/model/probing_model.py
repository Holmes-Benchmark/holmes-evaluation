from collections import defaultdict
from typing import Dict, List

import numpy
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from defs.schema import ProbingEntry


class SkeletonProbingModel(LightningModule):
    def __init__(self, hyperparameter, unique_inputs={}):
        super().__init__()
        self.hyperparameter = hyperparameter

        self.train_step_outputs = []
        self.dev_step_outputs = []
        self.test_step_outputs = []

        if self.hyperparameter["num_labels"] == 1:
            self.mean_loss = torch.nn.SmoothL1Loss()
            self.loss = torch.nn.SmoothL1Loss(reduction='none')
            self.metrics = {
                "pearson": torchmetrics.PearsonCorrCoef()
            }
        else:
            self.mean_loss = torch.nn.CrossEntropyLoss()
            self.loss = torch.nn.CrossEntropyLoss(reduction='none')

            task_type = "binary" if self.hyperparameter["num_labels"] == 2 else "multiclass"

            self.metrics = {
                "acc": torchmetrics.Accuracy(task=task_type),
                "f1": torchmetrics.F1Score(
                    average="macro", num_classes=self.hyperparameter["num_labels"], task=task_type
                ),
            }
            if self.hyperparameter["num_labels"] > 2:
                self.f1_all = torchmetrics.F1(average="none", num_classes=self.hyperparameter["num_labels"])
                self.acc_all = torchmetrics.Accuracy(average="none", num_classes=self.hyperparameter["num_labels"])

        self.best_val_metrics = defaultdict(lambda: -1)
        self.best_test_metrics = defaultdict(lambda: -1)
        self.test_preds = []
        self.unique_inputs = unique_inputs



    def batching_collate(self, batch:List):
        encoded_inputs = torch.FloatTensor(numpy.stack([element[1] for element in batch]))

        if self.hyperparameter["num_labels"] == 1:
            labels = torch.Tensor(
                [element[2] for element in batch]
            )
        else:
            labels = torch.LongTensor(
                [element[2] for element in batch]
            )

        return encoded_inputs, labels


    def test_batching_collate(self, batch:List[ProbingEntry]):
        encoded_inputs = torch.FloatTensor(numpy.stack([element[1] for element in batch]))

        if self.hyperparameter["num_labels"] == 1:
            labels = torch.Tensor(
                [element[2] for element in batch]
            )
        else:
            labels = torch.LongTensor(
                [element[2] for element in batch]
            )

        seen_indices = torch.BoolTensor(
            [
                element[0] for element in batch
            ]
        )

        return encoded_inputs, labels, seen_indices

    def get_dataloader(self, dataset, batch_size, shuffle=False):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
        dataloader.collate_fn = self.batching_collate
        return dataloader

    def get_test_dataloader(self, dataset, batch_size, shuffle=False):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
        dataloader.collate_fn = self.test_batching_collate
        return dataloader

    def run_val_step(self, batch, prefix):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self(x)
        if self.hyperparameter["num_labels"] > 1:
            pred = torch.nn.Softmax()(pred)
            losses = self.loss(pred, y)
        else:
            losses = self.loss(pred, y.unsqueeze(dim=1))
            pred = pred.squeeze(dim=1)
            losses = losses.squeeze(1)

        self.log(prefix + " loss", losses.mean(), on_epoch=True, prog_bar=True)
        self.dev_step_outputs.append([losses, pred, y])
        return losses, pred, y

    def run_test_step(self, batch, prefix):
        x, y, seen_indices = batch
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self(x)
        if self.hyperparameter["num_labels"] > 1:
            pred = torch.nn.Softmax()(pred)
            losses = self.loss(pred, y)
        else:
            losses = self.loss(pred, y.unsqueeze(dim=1))
            pred = pred.squeeze(1)
            losses = losses.squeeze(1)

        #self.log(prefix + " loss", losses.mean(), on_epoch=True, prog_bar=True)
        self.test_step_outputs.append([losses, pred, y, seen_indices])
        return losses, pred, y, seen_indices

    def training_step(self, batch, batch_index):
        x, y = batch[0]
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self(x)
        if self.hyperparameter["num_labels"] > 1:
            pred = torch.nn.Softmax()(pred)
            loss = self.mean_loss(pred, y)
        else:
            loss = self.mean_loss(pred, y.unsqueeze(dim=1))

        self.train_step_outputs.append(loss)

        return loss

    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(self.hyperparameter)

    def validation_step(self, batch, batch_index):
       # set = "val" if dataloader_index == 0 else "test"
        return self.run_val_step(batch, "val")

    def test_step(self, batch, batch_index):
        return self.run_test_step(batch, "test")


    def on_train_epoch_end(self):
        self.log("train loss", torch.tensor(self.train_step_outputs).mean())
        self.train_step_outputs.clear()

    def on_test_epoch_end(self):

        losses = [ele[0] for ele in self.test_step_outputs]
        losses = torch.concat(losses)
        losses_mean = losses.mean()
        losses_sum = losses.sum()

        preds = [ele[1] for ele in self.test_step_outputs]
        truths = [ele[2] for ele in self.test_step_outputs]

        seen_indices = [ele[3] for ele in self.test_step_outputs]

        pred_labels = torch.cat(preds).detach().cpu()
        truth_labels = torch.cat(truths).detach().cpu()
        seen_indices = torch.cat(seen_indices).detach().cpu()
        unseen_indices = seen_indices == False
        metric_results = {}

        lower_quantile = truth_labels.quantile(q=0.25)
        upper_quantile = truth_labels.quantile(q=0.75)

        lower_quantile_indices = (truth_labels < lower_quantile).nonzero().squeeze()
        upper_quantile_indices = (truth_labels > upper_quantile).nonzero().squeeze()

        for set_name, preds, labels in [
            ("full", pred_labels, truth_labels),
            ("seen",  pred_labels[seen_indices], truth_labels[seen_indices]),
            ("unseen",  pred_labels[unseen_indices], truth_labels[unseen_indices]),
            ("lower",  pred_labels[lower_quantile_indices], truth_labels[lower_quantile_indices]),
            ("upper",  pred_labels[upper_quantile_indices], truth_labels[upper_quantile_indices]),
        ]:
            if len(preds) == 0:
                continue

            for metric, func in self.metrics.items():
                #if metric == "pearson":
                #    pred_labels = pred_labels.squeeze(dim=1)
                if self.hyperparameter["num_labels"] > 1:
                    metric_result = func(pred_labels.argmax(1), truth_labels)
                else:
                    metric_result = func(pred_labels, truth_labels)
                metric_results[set_name + " test " + metric] = float(metric_result)

            if self.hyperparameter["num_labels"] > 2:
                for label, value in enumerate(self.f1_all.cpu()(preds.cpu(), labels.cpu())):
                    metric_results["z_" + set_name + " test f1-" + str(label)] = float(value)
                for label, value in enumerate(self.acc_all.cpu()(preds.cpu(), labels.cpu())):
                    metric_results["z_" + set_name + " test acc-" + str(label)] = float(value)

        for metric, result in metric_results.items():
            self.best_test_metrics[metric] = result
            self.log(metric, result, on_epoch=True, prog_bar=True)

        self.best_test_metrics["summed_loss"] = float(losses_sum.detach().cpu())
        if self.hyperparameter["num_labels"] >= 2:
            self.test_raw_preds = pred_labels.detach().cpu().double()
            self.test_preds = pred_labels.argmax(dim=1).detach().cpu().double().numpy()
            self.test_labels = truth_labels.detach().cpu()
            self.test_losses = losses.detach().cpu().double().numpy()
            self.test_seen_indices = seen_indices
        else:
            self.test_raw_preds = pred_labels.detach().cpu().double()
            self.test_preds = pred_labels.detach().cpu().double().numpy()
            self.test_labels = truth_labels.detach().cpu()
            self.test_losses = losses.detach().cpu().double().numpy()
            self.test_seen_indices = seen_indices

        self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.process_validation_results(self.dev_step_outputs)
        self.dev_step_outputs.clear()

    def process_validation_results(self, validation_step_outputs):
        losses = [ele[0] for ele in validation_step_outputs]
        losses = torch.concat(losses)

        preds = [ele[1] for ele in validation_step_outputs]
        truths = [ele[2] for ele in validation_step_outputs]

        pred_labels = torch.cat(preds).detach().cpu()
        truth_labels = torch.cat(truths).detach().cpu()

        summed_loss = losses.sum()

        metric_results = {}

        for metric, func in self.metrics.items():

            if self.hyperparameter["num_labels"] > 1:
                metric_result = func(pred_labels.argmax(1), truth_labels)
            else:
                metric_result = func(pred_labels, truth_labels)

            metric_results[metric] = metric_result

        if "pearson" in metric_results:
            ref_metric = float(metric_results["pearson"].detach().cpu())
        else:
            ref_metric = float(metric_results["f1"].detach().cpu())

        if ref_metric >= self.best_val_metrics["ref"] and self.current_epoch > 2:
            self.best_val_metrics["summed_loss"] = float(summed_loss.detach().cpu())
            self.best_val_metrics["ref"] = ref_metric
            self.best_val_metrics["epoch"] = ref_metric
                
            for metric, result in metric_results.items():
                self.best_val_metrics[metric] = float(result.detach().cpu())

        for metric, metric_result in metric_results.items():
            self.log("val " + metric, metric_result,  on_epoch=True, prog_bar=True)

        if self.hyperparameter["num_labels"] > 2:
            for label, value in enumerate(self.f1_all.cpu()(pred_labels.cpu(), truth_labels.cpu())):
                self.log("z_val f1-" + str(label), value, on_epoch=True, prog_bar=False)
            for label, value in enumerate(self.acc_all.cpu()(pred_labels.cpu(), truth_labels.cpu())):
                self.log("z_val acc-" + str(label), value,  on_epoch=True, prog_bar=False)

        self.log("val loss sum", summed_loss,  on_epoch=True, prog_bar=True)

        if self.hyperparameter["num_labels"] >= 2:
            self.dev_preds = pred_labels.argmax(dim=1).detach().cpu().int().numpy()
            self.dev_losses = losses.detach().cpu().double().numpy()
        else:
            self.dev_preds = pred_labels.detach().cpu().double().numpy()
            self.dev_losses = losses.detach().cpu().double().numpy()

        self.log_custom_metrics(metric_results, "val")

        self.log("val_ref", ref_metric)


    def configure_optimizers(self):
        optimizer = self.hyperparameter["optimizer"](self.parameters(), lr=self.hyperparameter["learning_rate"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )




class LinearProbingModel(SkeletonProbingModel):
    def __init__(self, hyperparameter: Dict, unique_inputs={}):

        super().__init__(hyperparameter, unique_inputs)


        self.layers = ModuleList([])
        self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

        input_dim = self.hyperparameter["input_dim"]
        if self.hyperparameter["num_hidden_layers"] > 0:
            for i in range(self.hyperparameter["num_hidden_layers"]):
                self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["hidden_dim"]))
                self.layers.append(torch.nn.LeakyReLU())
                self.layers.append(torch.nn.Dropout(self.hyperparameter["dropout"]))

                input_dim = self.hyperparameter["hidden_dim"]


        self.layers.append(torch.nn.Linear(input_dim, self.hyperparameter["num_labels"]))

    def forward(self, x:torch.Tensor):
        for l in self.layers:
            x = l(x)
        pred = x
        return pred


    def log_custom_metrics(self, metric_results, set):
        pass


