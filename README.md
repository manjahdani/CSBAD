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

The distillation + acquisition framework is presented in the following figure: 

![Pipeline](imgs/kd_pipeline.png)

<a name="requirements"></a>
## Requirements

First, setup your environment in command line:
```
conda create -n trail22 python=3.9
conda activate trail22
```


Once you have cloned this repository, you can install the requirements entering the following command:

```
git clone -b v6.2.1 https://github.com/Gerin-Benoit/yolov5.git
pip install -r yolov5/requirements.txt
pip install -r requirements.txt
pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

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