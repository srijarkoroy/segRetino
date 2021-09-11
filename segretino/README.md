## Model Architecture
![unet](https://user-images.githubusercontent.com/66861243/132897468-8004e34c-3637-4ced-8225-c1ad971b451b.png)

## Model Initialization
The UNET model for retina blood vessel segmentation may be initialized and the state dict may be viewed by running the following code snippet:

```python
from unet import UNET

net = UNET()
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
```

## Loss Functions

### Dice Loss
The Dice Loss Function may be initialized and called by running the following code snippet:

```python
from loss import DiceLoss

loss_fn = DiceLoss()
score, loss = loss_fn(output_var, target_var) #output_var is the output mask and target_var is the label
```
### Intersection over Union
The IoU Loss Function may be initialized and called by running the following code snippet:

```python
from loss import IoU

loss_fn = IoU()
score, loss = loss_fn(output_var, target_var) #output_var is the output mask and target_var is the label
```
## Model Training
We train the UNET model with Dice Loss and Intersection over Union as the loss functions and Adam as the optimizer with a learning rate of 1e-4 for 50 epochs. Since the images are large (512x512) we use a batch size of 1. 

The training object may be initialized and the unet model can be properly trained by running the following code snippet:
```python
from unet import UNET
from loss import DiceLoss, IoU
from training_utils import Train

from torch.optim import Adam

train = Train(dice = DiceLoss(), iou = IoU())

model = UNET()
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
train.forward((model = model, loader = <dataloader_object>, optimizer = optimizer, epoch = 50)
```
