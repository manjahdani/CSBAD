# trail22KD

## Requirements

### Working on Host (No Docker)
Once you have cloned this repository, you can install the requirements entering the following command:

```
pip install -r yolov8/ultralytics/requirements.txt
pip install -r requirements.txt
```

### Using Docker :


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
cd yolov8
python train.py --name S0XC0XX-firstn-300 --entity trail22kd --project kdtestv8 ....other arguments

# example : 
python train.py --name S05c016-firstn-100 --entity trail22kd --project kdtestv8 --epochs 10 --data  ../training/trail22kd.yaml --weights yolov8n.pt
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
docker image build -t fennecinspace/kdtestv8 --build-arg WANDB_API_KEY=1468bcb73576f41791fa4555a9ddd2b4a782ce38 .
```

PS: Do not have the data folder in this repo's directory, it will increase the image size when building.

#### run
The workdir in for this image is the `yolov8` directory. Please call your python files in a relative manner to that directory.

```
docker run --rm --gpus all -v /path/to/data/S05c016:/workspace/data fennecinspace/kdtestv8 python command...

# subsampling example
docker run --rm --gpus all -v /home/fennecinspace/dataset/S05c016:/workspace/data fennecinspace/kdtestv8 python ../strategy/select_strategy.py -n 100 -f "/workspace/data" -s flow_interval_mix

# training example
docker run --rm --gpus all -v /home/fennecinspace/dataset/S05c016:/workspace/data fennecinspace/kdtestv8 python train.py --name S05c016-firstn-100 --entity trail22kd --project kdtestv8 --epochs 10 --data /workspace/training/docker/trail22kd.yaml --weights yolov8n.pt
```

PS: you'll need to have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) support to have access to the GPU inside docker containers. 


## Other INFO

If you add parameters during training, make a note of it somewhere. For example if you use a batch number of 32 instead of the default 16, set your run name to : `S05c016-firstn-100-batch-8`