# trail22KD

## Installation

### Clone the repo
```bash
git clone https://github.com/manjahdani/trail22KD.git
cd trail22KD
```

### Create conda env
```bash
conda create -y --name trail22kd python pip
conda activate trail22kd
```

### Install requirements
```bash
pip install -r requirements.txt
```

### Configure wandb
- Check if you're in the ["trail22kd" team](https://wandb.ai/trail22kd)

- Setup wandb to send logs :
```bash
wandb login
```

## Dataset structure
Your dataset folder should follow the same structure:

### WALT

```
WALT-challenge
├── cam{1}
│   ├── week{1}
│   │   └── bank
│   │   │   ├── images
│   │   │   └── labels
|   |   └── test
│   │       ├── images
│   │       └── labels
.   .
│   └── week{i}
│       └── ...
.
└── cam{j}
    └── ...
```

### AI-city

```
AI-city
├── S01c011
│   ├── bank
│   │   ├── images
│   │   └── labels
│   ├── test
│   │   ├── images
│   │   └── labels
.
└── S{i}c{j}
    └── ...
```

## Running code

### Generation of the pseudo labels (populate bank)

```bash
python annotation/generate_pseudo_labels.py --parent "YOURPATH/WALT-or-AI-city/bank" --extension "jpg-or-png"
```
*remark*: bank folder must contain an images/ folder that contains all the images. If you are on Windows, you can you only use the `"` and **not** the `'`.

### Conduct an experiment

Running an experiment consists in the following steps:
    
1. Populate a `val` folder based on the data contained in bank  folder.
2. Populate a `train` folder based on a `strategy` applied on the data contained in bank folder. The `strategy` ensures that no images are duplicated between the train and the val sets.
3. Launch a training on a `yolov8n` model based on the previously generated sets. The scripts automatically launches everything to wandb.


The following is based on the configuration file of [Hydra](https://hydra.cc/) which is a powerful configuration tool which allow to modify easily the conducted experiments.

You can launch an experiment by executing main:

```bash
python main.py
```

If needed, you can add `HYDRA_FULL_ERROR=1` as env variable to see a bit more clearly the Traceback in case of debuging.

```bash
HYDRA_FULL_ERROR=1 python main.py
```

#### Modify the configs to change the experiments
To make a long story short, hydra is a configuration management tool that retrieves the information included in the `*.yaml` files to facilitate the deployment of the application.

To modify an experiment you can modify the configuration file `experiments/experiment.yaml`. **At your first use, you will have to modify the paths to the dataset and your wandb username.**

The logs, outputs and stuffs of your runs are automatically outputed in the `output` folder.

*remark*: if you are using Windows, do not forget to adapt your paths by using `/` instead of **not** `\` or `//`.


## Download, test and generate plots

- `--run-prefix` : wandb project name
- `--entity` : wandb project team
- `--project` : project name used as prefix in the runs names (sometimes it's different than the wandb project name, like in the case of "study")
- `--template` : template file for data.yaml used to specify test "sub"datasets
- `--dataset_path` : parent path containing all dataset folders (S0XC0XX in case of AI-City, camX in case of WALT) 

```
python test.py --run-prefix AI-city --entity trail22kd --project Ai-city --template testing/templates/Ai-city.yaml --dataset_path /mnt/ad2e7514-1ee8-465d-8ea9-010c3411d4e4/TRAIL/dataset
```

Results will be in `testdir/project`. `wandb` folder containing downloaded weights, `inference_results.csv` containing inference results, and `plots` folder containing generated plots. Modify plots look in `testing/plot.py`

You can also run script individually :

```
# download 
python testing/download.py -e trail22kd -p Ai-city -f ./testdir/Ai-city/wandb -lf -d

# inference
python3 ./testing/inference.py -w ./testing/Ai-city/wandb -d /mnt/ad2e7514-1ee8-465d-8ea9-010c3411d4e4/TRAIL/dataset -p AI-city -y testing/templates/Ai-city.yaml -f test -c ./testdir/Ai-city/inference_results.csv

# coco student inference (teacher: change --model yolov8n to yolov8x)
python3 testing/inference_coco.py --model yolov8n --csv_path ./testdir/Ai-city/inference_results.csv --dataset "s05c016->/mnt/ad2e7514-1ee8-465d-8ea9-010c3411d4e4/TRAIL/dataset/s05c016/" --dataset "s05c017->/mnt/ad2e7514-1ee8-465d-8ea9-010c3411d4e4/TRAIL/dataset/s05c017/" --data-template testing/templates/Ai-city.yaml --folder test

# plotting
python testing/plot.py --csv_path ./testdir/Ai-city/inference_results.csv --save_path ./testdir/Ai-city/plots
```

---
# Deprecated

### Using Docker (deprecated):

Clone the YoloV5 repo :

```
git clone -b v6.2.1 https://github.com/ultralytics/yolov5
```

Install [Docker](https://docs.docker.com/get-docker/) and [Docker Nvidia GPU Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

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