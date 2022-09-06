# trail22KD

## Requirements

First, setup you environment in command line:
```
conda create -n trail22 python=3.9
conda activate trail22
```

Once you have cloned this repository, you can install the requirements entering the following command:

```
git clone -b v6.2 https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
pip install -r requirements.txt
pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Training :

- Check if you're in the ["trail22kd" team](https://wandb.ai/trail22kd) Team

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