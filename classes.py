import ast

import cv2
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torchmetrics.classification import BinaryFBetaScore, Recall, Precision


class FaceNet(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = timm.create_model(
            'resnet18', pretrained=True, num_classes=model_hparams['num_classes'])
        
        # Create loss module
        self.loss_module = FocalLoss()
        self.valid_recall= Recall(task='binary')
        self.valid_precision= Precision(task='binary')
        self.test_recall = Recall(task='binary')
        self.test_precision = Precision(task='binary')
        self.example_input_array = torch.zeros(
            (1, 3, 512, 512), dtype=torch.float32)

    def forward(self, img):
        # Forward function that is run when visualizing the graph
        return self.model(img)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **
                                  self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        self.valid_recall.update(preds.argmax(dim=-1), labels)
        self.valid_precision.update(preds.argmax(dim=-1), labels)
        # By default logs it per epoch (weighted average over batches)
        self.log("valid_loss", loss)

    def validation_epoch_end(self, validation_step_outputs):
        valid_recall = self.valid_recall.compute()
        self.valid_recall.reset()
        self.log("valid_recall", valid_recall)

        valid_precision = self.valid_precision.compute()
        self.valid_precision.reset()
        self.log("valid_precision", valid_precision)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.test_recall.update(preds.argmax(dim=-1), labels)
        self.test_precision.update(preds.argmax(dim=-1), labels)
    def test_epoch_end(self, test_step_outputs):
        test_recall = self.test_recall.compute()
        self.test_recall.reset()
        self.log("test_recall", test_recall)

        test_precision = self.test_precision.compute()
        self.test_precision.reset()
        self.log("test_precision", test_precision)

class CustomDataset(Dataset):
    def __init__(self, data, augmentations=None):
        self.augmentations = augmentations
        self.images = [x.image for x in data]
        self.spoof_types = [x.spoof_type for x in data]
        self.spoof_labels = [x.spoof_label for x in data]
        self.boxes = [x.box for x in data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        box = ast.literal_eval(self.boxes[index])
        image = cv2.imread(str(self.images[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[box[1]:box[3], box[0]:box[2]]
        spoof_label = self.spoof_labels[index]

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']/255
            return image, spoof_label
        else:
            return image/255, spoof_label


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.example_input_array = torch.zeros(
            (60, 2), dtype=torch.float32)

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        out = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
        return out.mean()
