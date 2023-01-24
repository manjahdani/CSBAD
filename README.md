# trail22KD

## Requirements

First, setup you environment in command line:
```
conda create -n trail22 python=3.9
conda activate trail22
```

Once you have cloned this repository, you can install the requirements entering the following command:

```
pip install ultralytics
pip install -r requirements.txt
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