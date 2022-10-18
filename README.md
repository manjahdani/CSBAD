# Data acquisition with knowledge distillation for context-aware large-scale surveillance

This repository contains the source code produced by the team [??] during the 2022 TRAIL summer workshop 
which took place in Berlin.

The project objectives are available here for additional background and information.

## Table of Contents
1. [Overall description](#description)
2. [Requirements and Installation](#requirements)
3. [Usage](#usage)
4. [Acknowledgements](#ack)

<a name="description"></a>
## Overall description
This repository implements a pipeline of data acquisition and _teacher-student_ knowledge 
distillation framework as presented in the following figure.

![Pipeline](imgs/kd_pipeline.png)

The goal is to specialize a lightweight deep-learning detector model to its particular viewpoint.
It allows the model to be embedded at the sensor level while reaching at least the same performances
as a heavier general-purpose detector trained on large dataset (e.g., COCO). The heavier model (AKA the _teacher_) is used as a pseudo annotator for the lightweight model (the _student_).
Forwarding a set of frames through the _teacher_ will produce a pseudo annotated set of images which can be split
in a training and a validation set.

Besides, the other objective is to study how to decrease the number of frames the camera has to send to the central server
to produce an efficient specialized lightweight model in order to decrease bandwidth usage. Several sampling strategies
are implemented and explained in the folder `/strategy` README.


<a name="requirements"></a>
## Requirements and installation

First, set up a Python 3.9 environment. The simplest is to use `conda` in command line:
```
conda create -n trail22 python=3.9
conda activate trail22
```


Once you have cloned this repository, you can install the requirements entering the following command:

```
cd trail22DK/
git clone -b v6.2.1 https://github.com/Gerin-Benoit/yolov5.git
pip install -r yolov5/requirements.txt
pip install -r requirements.txt
pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Note that we are using a slightly modified version of the excellent open-source [Ultralytics YOLOv5 🚀](https://github.com/ultralytics/yolov5) to train, validate and
infer YOLOv5 models.


### Weights and Bias monitoring

- Check if you're in the ["trail22kd" team](https://wandb.ai/trail22kd) team

- Setup wandb to send logs :
```
pip install wandb
wandb login
```
PS: for login use the key found in the ["trail22kd" team](https://wandb.ai/trail22kd) team page.

- Train : the name of the run should be S0XC0XX-STRATEGY-NUMBER_OF_FRAMES :
```
python yolov5/train.py --name S0XC0XX-firstn-300 --project kd --entity trail22kd ....other arguments
```
<a name="usage"></a>
## Usage

<a name="ack"></a>
## Acknowledgements