import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import datetime
from model import MoleculeRegressor  # import your model from wherever it’s defined
import os


class MoleculeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def train_and_predict(X_train, y_train, X_val, y_val, X_test, gpu=6):
    # Create dataloaders
    num_workers = os.cpu_count() // 2 
    train_loader = DataLoader(MoleculeDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(MoleculeDataset(X_val, y_val), batch_size=32, num_workers=num_workers)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Define model
    model = MoleculeRegressor(input_size=X_train.shape[1])

    # Callbacks
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    checkpoint_path = f'/home/malves/predinhib_mtb/models/best_model_tb_reg_{time_stamp}.ckpt'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/home/malves/predinhib_mtb/models/',
        filename=f'best_model_tb_reg_{time_stamp}',
        save_top_k=1,
        mode='min',
        save_weights_only=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='gpu',
        devices=[gpu],
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best model
    best_model = MoleculeRegressor.load_from_checkpoint(checkpoint_path, input_size=X_train.shape[1])
    best_model.eval().to(f"cuda:{gpu}")

    with torch.no_grad():
        test_tensor = test_tensor.to(f"cuda:{gpu}")
        y_pred = best_model(test_tensor).cpu().numpy().flatten()

    return y_pred
