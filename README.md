# That is implementation of Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots for SBER Robotics Lab

article: [Blind2Unblind](https://arxiv.org/abs/2203.06967)

## Citing Blind2Unblind
```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zejin and Liu, Jiazheng and Li, Guoqing and Han, Hua},
    title     = {Blind2Unblind: Self-Supervised Image Denoising With Visible Blind Spots},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2027-2036}
}
```

The original code is placed here: [github](https://github.com/zejinwang/Blind2Unblind)
## Installation
The model is built in Python3.8.5, PyTorch 1.7.1 in Ubuntu 22.04 environment.

## Data Preparation

### 1. Prepare Training Dataset

Please put your training dataset under the path: **./b2u_sber_implemetation/data/train**.

<!-- - For processing ImageNet Validation, please run the command

  ```shell
  python ./dataset_tool.py
  ```

- For processing SIDD Medium Dataset in raw-RGB, please run the command

  ```shell
  python ./dataset_tool_raw.py
  ``` -->

### 2. Prepare Validation Dataset

​	Please put your validation dataset under the path: **./b2u_sber_implemetation/data/test**.

## Pretrained Models
You can download pre-trained models here: [Google Drive](https://drive.google.com/drive/folders/1ruA6-SN1cyf30-GHS8w2YD1FG-0A-k7h?usp=sharing) 


The pre-trained models are placed in the folder: **./b2u_sber_implemetation/pretrained_models**


Models were trained on datasets G-209, Crystal_focus_0_dose_180, G-146


```yaml
# # For more noisy datasets processing use model firstly trained on G-209

./pretrained_models/b2u_first.pth
# Than use model secondly trained on G-209 denoised by first model
./pretrained_models/b2u_second.pth


# # For less noisy images use model trained on Crystal_focus_0_dose_180
./pretrained_models/b2u_crystal_first.pth
```

## Train
* For training your own model please use [SBER_train](https://github.com/sadevans/b2u_sber_implemetation/blob/f2865e86ba95634329dfbdb229182295d3da0425/SBER_train.ipynb#L10)


## Test

Please put your test data in the folder: **./b2u_sber_implemetation/test**

* To test model on images maximum size **768x1024** use [SBER_test_small_images](https://github.com/sadevans/b2u_sber_implemetation/blob/main/SBER_test_small_images.ipynb)

* To test model on large resolution images use [SBER_test_large_images](https://github.com/sadevans/b2u_sber_implemetation/blob/main/SBER_test_large_images.ipynb)

In this jupyter notebook you can set: 
- your image proportions, 
- crop propotions
- margin value for cropping and concating without visible joints