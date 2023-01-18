# trail22KD

## Requirements

### Working on Host (No Docker)
Once you have cloned this repository, you can install the requirements entering the following command:

```
git clone -b v6.2.1 https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
pip install -r requirements.txt
pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Using Docker :

Clone the YoloV5 repo :

```
git clone -b v6.2.1 https://github.com/ultralytics/yolov5
```

Install [Docker](https://docs.docker.com/get-docker/) and [Docker Nvidia GPU Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)


## Running code

### Subsampling

```
python strategy/select_strategy.py -n 100 -f "path/to/data/S05c016" -s fixed_interval
```

Strategies : n_first, random, fixed_interval, flow_diff, flow_interval_mix


### Training

- Check if you're in the ["trail22kd" team](https://wandb.ai/trail22kd) Team

- Setup wandb to send logs :
```
pip install wandb
wandb login
```
PS: for login use the API key found in your [WANDB SETTINGS](https://wandb.ai/settings) page.

- Train : the name of the run should be S0XC0XX-STRATEGY-NUMBER_OF_FRAMES :
```
python yolov5/train.py --name S0XC0XX-firstn-300 --project kdtest --entity trail22kd ....other arguments

# example : 
python yolov5/train.py --name S05c016-firstn-100 --project kdtest --entity trail22kd --data  training/trail22kd.yaml --hyp training/hyp.trail22kd.yaml --weights models/yolov5n.pt
```


### Testing

- manage trained weights

```
python testing/download.py
# example : download all finished runs to downloads folder 
python testing/download.py -f downloads -lf -d
```

- Test all downloaded models :

```
python testing/inference.py -w downloads -c testing-results.csv --d path/to/data --classes 0
```

## Running code with Docker (GPU) (NO NEED FOR VIRTUAL ENV & CLONING)
You can execute the subsampling and training commands through docker :

#### build 

```
docker image build -t fennecinspace/kdtest --build-arg WANDB_API_KEY=YOUR_WANDB_API_KEY .
```

PS: Do not have the data folder in this repo's directory, it will increase the image size when building.

#### run

```
docker run --rm --gpus all -v /path/to/data/S05c016:/workspace/data fennecinspace/kdtest python command...

# subsampling example
docker run --rm --gpus all -v /home/fennecinspace/dataset/S05c016:/workspace/data fennecinspace/kdtest python strategy/select_strategy.py -n 100 -f "/workspace/data" -s flow_interval_mix

# training example
docker run --rm --gpus all -v /home/fennecinspace/dataset/S05c016:/workspace/data fennecinspace/kdtest python yolov5/train.py --name S05c016-fixed_interval-100 --project kdtest --entity fennecinspace --data  /workspace/training/docker/trail22kd.yaml --hyp /workspace/training/hyp.trail22kd.yaml --weights /workspace/models/yolov5n.pt
```

PS: you'll need to have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) support to have access to the GPU inside docker containers. 