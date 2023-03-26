# Conditional-Diffusion-models
This repository implement conditional diffusion model from scratch and train it on the [MNIST-M](https://www.kaggle.com/datasets/yuna1117/mnist-m) dataset. Given conditional labels 0-9, and generate the corresponding digit images.

## MNIST-M Dataset
Generated from MNIST<br>
\# of data: 44,800 / 11,200 (training/validation)<br>
\# of classes: 10 (0~9)<br>
A subset of MNIST - The digit images are normalized (and centered) in size 28 * 28 * 3 pixels

## Training
```
python train.py --data_root $data_directory
```
## Download checkpoint
[checkpoint link](https://drive.google.com/drive/u/0/folders/1dZtdGKg-caE4FOcLbdKiBL7gu-ssQeY_)
```
bash download.sh
```
## Sampling Images
```
python sampling.py --out_dir $output_directory --checkpoint $checkpoint_directory
```
<p align="center">
<img src="https://drive.google.com/uc?id=1VmQEtoZCD16itdfPoiDe0OVDatv_Pv0h"/>
</p>
<p align="center">
Fig 1: (0-9) generate images
</p>
<p align="center">
<img src="https://drive.google.com/uc?id=1astGBIDINJPUAdt2lcr50OD-IpMkRAye" width="12%" hspace="8"/>
<img src="https://drive.google.com/uc?id=1mANNoXhTMsK9Ft9KbA0cgMCP8AFl5-1G" width="12%" hspace="8"/>
<img src="https://drive.google.com/uc?id=1qiffugCcMcCxYuCBwaNwhQWtpqNfnNND" width="12%" hspace="8"/>
<img src="https://drive.google.com/uc?id=14t0eeljz1jCx0UsuAMtE9yPonq3fk3Yu" width="12%" hspace="8"/>
<img src="https://drive.google.com/uc?id=1l-_7qNeXsnC1io_9v5drg7T3C5-uQhh7" width="12%" hspace="8"/>
<img src="https://drive.google.com/uc?id=1ds14GWektmlm895Q8queh5q44GqbiCjp" width="12%" hspace="8"/>
</p>
<p align="center">
(a) t=0 &emsp;&emsp;&emsp;(b) t=80 &emsp;&emsp;&emsp;(c) t=160 &emsp;&emsp;&emsp;(d) t=240 &emsp;&emsp;&emsp;(e) t=320 &emsp;&emsp;&emsp;(f) t=400
</p>
<p align="center">
Figure 10: First ’0’ in different time steps
</p>
