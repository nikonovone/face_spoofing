import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from classes import FaceNet
from pytorch_lightning.loggers import TensorBoardLogger


def visual(img, bgr2rgb=False, gray=True):

    if isinstance(img, str):
        img = io.imread(img)
        plt.imshow(img)
        plt.show()

    elif isinstance(img, np.ndarray):
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if gray:
            plt.imshow(img, cmap='Greys_r')
        else:
            plt.imshow(img)
        plt.show()


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """

    if save_name is None:
        save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    logger = TensorBoardLogger(
        save_dir=kwargs['checkpoint_path'], log_graph=True, sub_dir='tb_logs')
    trainer = pl.Trainer(
        default_root_dir=os.path.join(
            kwargs['checkpoint_path'],save_name),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator='gpu' if str(kwargs['device']) == "cuda" else 'cpu',
        devices=kwargs['gpu_id'],
        # How many epochs to train for if no patience is set
        max_epochs=kwargs['max_epochs'],
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="valid_loss",
                # every_n_epochs = 1,
                # save_top_k = -1
            ), 
            LearningRateMonitor("epoch"),
        ],
        logger=logger,
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        kwargs['checkpoint_path'], 'lightning_logs/version_0/checkpoints/' + "model.ckpt")
    print(pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = FaceNet.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(kwargs['seed'])  # To be reproducable
        model = FaceNet(model_name=model_name, model_hparams=kwargs['model_hparams'],
                        optimizer_name=kwargs['optimizer_name'], optimizer_hparams=kwargs['optimizer_hparams'])
        trainer.fit(model, kwargs['train_loader'], kwargs['valid_loader'])
        model = FaceNet.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    valid_result = trainer.test(
        model, dataloaders=kwargs['valid_loader'], verbose=False)
    test_result = trainer.test(
        model, dataloaders=kwargs['test_loader'], verbose=False)
    result = {"test_recall": test_result[0]["test_recall"],
              "test_precision": test_result[0]["test_precision"],
              "valid_recall": valid_result[0]["test_recall"],
              "valid_precision": valid_result[0]["test_precision"]}
    logger.finalize("success")

    return model, result

# IN PROGRESS
# def test_model(model_name, **kwargs):


#     model = FaceNet(model_name=model_name, model_hparams=kwargs['model_hparams'],
#                     optimizer_name=kwargs['optimizer_name'], optimizer_hparams=kwargs['optimizer_hparams'])
#     model.load_from_checkpoint(
#         checkpoint_path=kwargs['checkpoint_path'],
#         hparams_file=kwargs['checkpoint_path'].parents[1] /
#         'tb_logs/hparams.yaml',
#     )
 
#     model.eval().to(kwargs['device'])
#     for x in model.parameters():
#         print(x)
#     preds = []
#     targets = []
#     imgs = []

#     for image, label in kwargs['test_loader']:
#         image = image[0][np.newaxis, ...].to(kwargs['device'])
#         predict = model(image).argmax(dim=-1)
#         predict = predict.detach().cpu().numpy()
#         preds.append(predict.item())
#         imgs.append(image)
#         targets.append(label.item())
#     return preds, targets
