# Documentation
This is the Documentation of segRetino, mentioning all Hyperparameters, Losses and Results.

### Parameters and Hyperparameters

image_dims | batch_size | optimizer | learning_rate | epochs 
:---: | :-----: | :------: | :------: | :------: |
512x512 | 1  | Adam | 1e-4 | 50 |

### Epoch-wise Loss progression
Epoch | Dice Loss | IoU Loss |
:----------: | :-----------: | :-----------: |
1 | 0.4024 | 0.5730
10 | 0.2370 | 0.3828
20 | 0.1735 | 0.2955
30 | 0.1437 | 0.2511 
40 | 0.1314 | 0.2320
50 | 0.1247 | 0.2215 

### Results
Original Image | Masked Image | Blend Image |
:-------------: | :---------: | :-----: |
<img src="/results/input/input1.png" height=200 width=200>| <img src="/results/output/output1.png" height=200 width=200>| <img src="/results/blend/blend1.png" height=200 width=200> |
<img src="/results/input/input2.png" height=200 width=200>| <img src="/results/output/output2.png" height=200 width=200>| <img src="/results/blend/blend2.png" height=200 width=200> |
<img src="/results/input/input3.png" height=200 width=200>| <img src="/results/output/output3.png" height=200 width=200>| <img src="/results/blend/blend3.png" height=200 width=200> |
