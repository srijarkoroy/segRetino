import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn

from segretino.unet import UNET
from segretino.loss import DiceLoss, IoU
from segretino.training_utils import Train, Evaluate

from dataset import DriveDataset

train_x = sorted(glob( < path / to / augmented / train / image / folder / >))
train_y = sorted(glob( < path / to / augmented / mask / image / folder / >))

valid_x = sorted(glob( < path / to / test / image / folder / >))
valid_y = sorted(glob( < path / to / test / mask / folder / >))

# Dataloader
train_dataset = DriveDataset(train_x, train_y)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2
)
val_dataset = DriveDataset(valid_x, valid_y)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

# Training and Evaluate object
train = Train(dice=DiceLoss(), iou=IoU())
eval = Evaluate(dice=DiceLoss(), iou=IoU())

# Model Initialization and setting up hyperparameters
model = UNET()
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 50

# Training
for epoch in tqdm(range(epochs)):
    print("Epoch: ", epoch)

    train_dice, train_iou = train.forward(model=model, loader=train_loader, optimizer=optimizer)
    val_dice, val_iou = eval.forward(model=model, loader=val_loader)

torch.save(model.state_dict(), 'unet.pth')
