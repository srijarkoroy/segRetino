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
loss_fn = DiceLoss()
score, loss = loss(output_var, target_var) #output_var is the output mask and target_var is the label
```
### Intersection over Union
The IoU Loss Function may be initialized and called by running the following code snippet:

```python
loss_fn = IoU()
score, loss = loss(output_var, target_var) #output_var is the output mask and target_var is the label
```
