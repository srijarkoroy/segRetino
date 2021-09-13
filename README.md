# Retina Blood Vessels Segmentation
This is an implementation of the research paper <a href = "https://researchbank.swinburne.edu.au/file/fce08160-bebd-44ff-b445-6f3d84089ab2/1/2018-xianchneng-retina_blood_vessel.pdf">"Retina Blood Vessel Segmentation Using A U-Net Based Convolutional Neural Network"</a> written by Wang Xiancheng, Li Weia, *et al.*

## Inspiration
Various eye diseases can be diagnosed through the characterization of the retinal blood vessels. The characterization can be extracted by using proper imaging techniques and data analysis methods. In case of eye examination, one of the important tasks is the retinal image segmentation.The paper presents a network and training strategy that relies on the data augmentation to use the available annotated samples more efficiently, to segment retinal blood vessels using a UNET convolutional neural network.

## Dataset
We have used the <a href = "https://drive.grand-challenge.org/">Digital Retinal Images for Vessel Extraction (DRIVE)</a> dataset for retinal vessel segmentation.
It consists of a total of JPEG 40 color fundus images; including 7 abnormal pathology cases. Each image resolution is 584x565 pixels with eight bits per color channel (3 channels), resized to 512x512 for our model.  

### Guidelines to download, setup and use the dataset
The DRIVE dataset may be downloaded <a href = "https://drive.google.com/file/d//view?usp=sharing">here</a> as a file named *dataset.zip*. 

**Please write the following commands on your terminal to extract the file in the proper directory**
```bash
  $ mkdir drive
  $ unzip </path/to/dataset.zip> -d </path/to/drive>
```
The resulting directory structure should be:
```
/path/to/drive
    -> dataset
        -> train
            -> image
                -> 21_training_0.tif
                -> 22_training_0.tif
                   ...
            -> mask
                -> 21_training_0.gif
                -> 22_training_0.gif
        -> test
            -> image
                -> 01_test_0.tif
                -> 02_test_0.tif
                   ...
            -> mask
                -> 01_test_0.gif
                -> 02_test_0.gif
```

## Model Architecture
The UNET CNN architecture may be divided into the *Encoder*, *Bottleneck* and *Decoder* blocks, followed by a final segmentation output layer. 

- Encoder: There are 4 Encoder blocks, each consisting of a convolutional block followed by a Spatial Max Pooling layer. 
- Bottleneck: The Bottleneck consists of a single convolutional block.
- Decoder: There are 4 Decoder blocks, each consisting of a deconvolution operation, followed by a convolutional block, along with skip connections.

**Note**: The *convolutional block* consists of 2 conv2d operations each followed by a BatchNorm2d, finally followed by a ReLU activation.

![model_arch](https://user-images.githubusercontent.com/66861243/133101290-eff181eb-bd9b-47cd-94b7-493d5c113dc0.png)

## Implementation Details
- Image preprocessing included augmentations like HorizontalFlip, VerticalFlip, Rotate.
- Dataloader object was created for both training and validation data
- Training process was carried out for 50 epochs, using the Adam Optimizer with a Learning Rate 1e-4.
- Validation was carried out using DiceLoss and Intersection over Union Loss. 
