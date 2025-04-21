import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MoleculeRegressor(pl.LightningModule):
    def __init__(self, input_size=2304, output_size=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, output_size)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            ),
            'monitor': 'val_loss'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        
        # Skip batch if size is 1
        if batch_size == 1:
            print(f"Skipping batch {batch_idx} in training with batch size 1.")
            return None  # Skip this batch
        
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        
        # Skip batch if size is 1
        if batch_size == 1:
            print(f"Skipping batch {batch_idx} in validation with batch size 1.")
            return None  # Skip this batch
        
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log('val_loss', loss)
        return loss