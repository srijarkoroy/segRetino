import os
import time
from glob import glob
 
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from .loss import DiceLoss, IoU
from .unet import UNET

class Train(nn.Module):

    def __init__(self, dice, iou):

        super().__init__()
        
        """
        This class is used for Training a UNET model.
        
        Parameters:

        - dice: DiceLoss object

        - iou: IoU object

        """

        self.loss_fn1 = dice
        self.loss_fn2 = iou
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, model, loader, optimizer):

        """
        Parameters:

        - model: Model Object/Unet Object

        - loader: DataLoader Object

        - optimizer: optimizer Object

        """

        model.train()

        if torch.cuda.is_available():
             print("Shifting the model to cuda!")
             model.cuda()

        epoch_loss1 = 0.0
        epoch_loss2 = 0.0

        #with tqdm(train_loader, unit="batch") as tepoch:

        for x,y in loader:

            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)

            score1,loss1 = self.loss_fn1(y_pred, y)
            score2,loss2 = self.loss_fn2(y_pred, y)

            loss1.backward(retain_graph = True)
            loss2.backward(retain_graph = True)

            optimizer.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

            #tepoch.set_postfix(loss=epoch_loss1, accuracy=100. * accuracy)
            #tepoch.set_postfix(loss=epoch_loss2, accuracy=100. * accuracy)

        epoch_loss1 = epoch_loss1/len(loader)
        epoch_loss2 = epoch_loss2/len(loader)

        print("Train Dice Loss: {}, ".format(epoch_loss1),"Train IoU Loss: {}, ".format(epoch_loss2))

        return epoch_loss1, epoch_loss2



class Evaluate(nn.Module):

    def __init__(self, dice, iou):

        super().__init__()
        
        """
        This class is used for Evaluating a UNET model.
        
        Parameters:

        - dice: DiceLoss object

        - iou: IoU object

        """

        self.loss_fn1 = dice
        self.loss_fn2 = iou
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, model, loader):

        """
        Parameters:

        - model: Model Object/Unet Object

        - loader: DataLoader Object

        """

        model.eval()

        if torch.cuda.is_available():
             print("Shifting the model to cuda!")
             model.cuda()


        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        
        with torch.no_grad():
            for x,y in loader:

                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = model(x)

                score1,loss1 = self.loss_fn1(y_pred, y)
                score2,loss2 = self.loss_fn2(y_pred, y)

                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()

            epoch_loss1 = epoch_loss1/len(loader)
            epoch_loss2 = epoch_loss2/len(loader)

            print("\nValidation Dice Loss: {}, ".format(epoch_loss1),"Validation IoU Loss: {}, ".format(epoch_loss2))

        return epoch_loss1, epoch_loss2
