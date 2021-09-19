## Data Loading
The ```augment.py``` file contains a function for loading the DRIVE Dataset, which can be called using the following code snippet:
```python
from augment import load_data

(train_x, train_y), (test_x, test_y) = load_data(<path/to/data/folder/>)
```

## Data Augmentation
Data Augmentation has been accomplished by doing **HorizontalFlip**, **VerticalFlip**, **Rotate**. 

The following code snippet shows the function call for data augmentation:
```python
from augment import augment_data

augment_data(train_x, train_y, <path/to/train/folder/>, augment=True)
augment_data(test_x, test_y, <path/to/validation/folder/>, augment=False)
```

## Creating the DataLoader 
The DataLoader object for training and validation data may be created by running the following code snippet:
```python
from dataset import DriveDataset

train_x = sorted(glob(<path/to/augmented/train/image/folder/>))
train_y = sorted(glob(<path/to/augmented/mask/image/folder/>))

valid_x = sorted(glob(<path/to/test/image/folder/>))
valid_y = sorted(glob(<path/to/test/mask/folder/>))

train_dataset = DriveDataset(train_x, train_y)
 
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
val_dataset = DriveDataset(valid_x,valid_y)

val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
```
