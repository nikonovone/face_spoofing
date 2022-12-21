import argparse
import os
from pathlib import Path

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from classes import CustomDataset
from utils import *


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_dir', default='celeba', type=str,
                        help='Path to train/test image folders and train/test .csv files')
    parser.add_argument('--check_dir', default='checkpoints',
                        type=str, help='Path to save logs, weights and etc.')
    parser.add_argument('--batch_size', default=60,
                        type=str, help='Batch size')
    parser.add_argument('--num_epochs', default=20,
                        type=int, help='Num epochs training')
    parser.add_argument('--num_workers', default=4,
                        help='How many subprocesses to use for data loading.')
    parser.add_argument('--lr', default=0.0001,
                        help='Learining rate')
    parser.add_argument('--gpu_id', default=0, type=str, help='Id GPU')
    opt = parser.parse_args()
    return opt


def main(opt):

    # PATHS
    DATASET_PATHS = Path(opt.data_dir)
    CHECKPOINT_PATH = Path(opt.check_dir)
    BATCH_SIZE = int(opt.batch_size)
    LEARNING_RATE = float(opt.lr)
    NUM_WORKERS = int(opt.num_workers)
    NUM_EPOCHS = int(opt.num_epochs)
    GPU_ID = int(opt.gpu_id)
    SEED = 13

    # READ DATA FROM DATASET FOLDER
    train_df = pd.read_csv(DATASET_PATHS / 'train.csv')
    test_df = pd.read_csv(DATASET_PATHS / 'test.csv')

    train_data = []
    test_data = []

    for x in train_df.itertuples():
        train_data.append(x)
    for x in test_df.itertuples():
        test_data.append(x)

    print(f'Train total len: {len(train_data)}',
        f'Test total len: {len(test_data)}', sep='\n')

    train, val = train_test_split(train_data, test_size=0.2, random_state=SEED)
    test = test_data

    print(f'Train shape: {len(train)}',
        f'Validation shape: {len(val)}',
        f'Test shape: {len(test)}', sep='\n')

    pl.seed_everything(SEED)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")


    # CREATE DATASETS, DATALOADERS
    train_transform = A.Compose([
        A.RandomGamma(),
        A.HorizontalFlip(p=0.5),
        A.Rotate(always_apply=False, limit=(-10, 10)),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(512, 512),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(512, 512),
        ToTensorV2(),
    ])

    train_dataset = CustomDataset(train, train_transform)
    valid_dataset = CustomDataset(val, test_transform)
    test_dataset = CustomDataset(test, test_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

    # TRAINING MODEL
    facenet_model, facenet_results = train_model(
        model_name="FaceNet",
        save_name='AugmentationAdd',
        model_hparams={"num_classes": 2},
        optimizer_name="Adam",
        optimizer_hparams={"lr": LEARNING_RATE},
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        gpu_id=[GPU_ID],
        max_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        seed=SEED,
    )

    test_loader = DataLoader(test_dataset, batch_size=1,
                             drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_model = facenet_model
    test_model.eval().to(device)

    preds = []
    targets = []
    imgs = []

    for image, label in test_loader:
        image = image[0][np.newaxis, ...].to(device)
        predict = test_model(image).argmax(dim=-1)
        predict = predict.detach().cpu().numpy()
        preds.append(predict.item())
        imgs.append(image)
        targets.append(label)

    print(f'f1_score:{f1_score([x.item() for x in targets], preds)}')
    print(f'precision:{precision_score([x.item() for x in targets], preds)}')
    print(f'recall:{recall_score([x.item() for x in targets], preds)}')


if __name__ == '__main__':
    opt = get_parse()
    main(opt)
