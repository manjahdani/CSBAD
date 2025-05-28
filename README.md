# Camera clustering for scalable stream-based active distillation

This repository contains the link to test data as well as the code to replicate the experiments in ``Camera clustering for scalable stream-based active distillation". Our methodology and full project scope is detailed in our paper [paper](https://arxiv.org/abs/2404.10411). Some additional informations can be found in our Supplementary Materials document [https://drive.google.com/file/d/1Jutgzw-nT0-8b_B2lO-XXDmF8jjsljMM/view?usp=sharing](https://drive.google.com/file/d/1Jutgzw-nT0-8b_B2lO-XXDmF8jjsljMM/view?usp=sharing). You can also find a brief link below. 


![Pipeline](images/SBAD-transparent.png)

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Testing](#testing)
- [FAQs](#faqs)

---

# MATERIALS - WALT Dataset

The WALT Watch And Learn 2D Amodal Representation using Time-lapse Imagery (WALT) dataset is available through Carnegie Mellon University's official website at https://www.cs.cmu.edu/~walt/. Please note that accessing the full dataset requires acceptance of their license terms. Our group offers additional annotations solely for subsets of images from the test set used in our methods. We have secured a written agreement from Carnegie Mellon for this purpose. All uses of the dataset, including the additional annotations, must adhere to the licensing terms specified by Carnegie Mellon. 

The WALT, captured by nine distinct cameras, offers a wide range of environmental settings. Figure below  showcases each camera’s viewpoint, highlighting the variety in visual conditions and challenges pertinent
to object detection tasks.

<table>
  <tr>
    <td align="center"><img src="images/cameras/cam1-min.jpg" width="250" height="150"><br /><sub>Cam 1</sub></td>
    <td align="center"><img src="images/cameras/cam2-min.jpg" width="250" height="150"><br /><sub>Cam 2</sub></td>
    <td align="center"><img src="images/cameras/cam3-min.jpg" width="250" height="150"><br /><sub>Cam 3</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/cameras/cam4-min.jpg" width="250" height="150"><br /><sub>Cam 4</sub></td>
    <td align="center"><img src="images/cameras/cam5-min.jpg" width="250" height="150"><br /><sub>Cam 5</sub></td>
    <td align="center"><img src="images/cameras/cam6-min.jpg" width="250" height="150"><br /><sub>Cam 6</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/cameras/cam7-min.jpg" width="250" height="150"><br /><sub>Cam 7</sub></td>
    <td align="center"><img src="images/cameras/cam8-min.jpg" width="250" height="150"><br /><sub>Cam 8</sub></td>
    <td align="center"><img src="images/cameras/cam9-min.jpg" width="250" height="150"><br /><sub>Cam 9</sub></td>
  </tr>
</table>

# MATERIALS - Test Set Details for the WALT Dataset

A subset of 1850 images was meticulously labeled to identify "vehicle" instances, resulting in 12,577 annotated instances. Table below summarizes the labeled dataset, providing insights into the annotation process and offering links to Roboflow for easy access to the data.
To access the dataset, append the specified path to this preamble: `https://universe.roboflow.com/sbad-dvvax`.

| CAM | ROBOFLOW Link |
|-----|---------------|
| 1   | [sbad_cam1_test_set/dataset/3](https://universe.roboflow.com/sbad-dvvax/sbad_cam1_test_set/dataset/3) |
| 2   | [sbad_cam2_test_set/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam2_test_set/dataset/1) |
| 3   | [sbad_cam3_test_set/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam3_test_set/dataset/1) |
| 4   | [sbad_cam4_test_set/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam4_test_set/dataset/1) |
| 5   | [sbad_cam5_test_set/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam5_test_set/dataset/1) |
| 6   | [sbad_cam6_test_set/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam6_test_set/dataset/1) |
| 7   | [sbad_cam7_test_set/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam7_test_set/dataset/1) |
| 8   | [sbad_cam8_test_set/dataset/2](https://universe.roboflow.com/sbad-dvvax/sbad_cam8_test_set/dataset/2) |
| 9   | [sbad_cam9_test_set-akhti/dataset/1](https://universe.roboflow.com/sbad-dvvax/sbad_cam9_test_set-akhti/dataset/1) |


# MATERIALS for Confirmation Bias 

In our experiment, in section VI.B, "*Impact of Teacher size*", we manually annotated **96** images sampled by a **YOLOv8n^COCO** *Student* utilizing the `Least-Confidence` or `Top-Confidence` approach and subsequently compared the performance of these *Students* against scenarios in which a *Teacher* model pseudo-annotated the same set of images and a human. In Table [tab:roboflowLinks], we present the images sampled by each strategy, which we annotated, thereby creating the equivalent human annotations.


### Table: Camera data strategies with corresponding Roboflow links
URLs are relative to the preamble: [https://universe.roboflow.com/sbad-dvvax/](https://universe.roboflow.com/sbad-dvvax/).

| Camera_id | Week | Strategy         | Roboflow link                                                                                                      |
|-----------|------|------------------|--------------------------------------------------------------------------------------------------------------------|
| 1         | 1    | `Top-Confidence` | [cam1_confbias_topconf96/dataset/3](https://universe.roboflow.com/sbad-dvvax/cam1_confbias_topconf96/dataset/3)    |
| 1         | 1    | `Least-Confidence` | [cam1_confbias_leastconf96/dataset/3](https://universe.roboflow.com/sbad-dvvax/cam1_confbias_leastconf96/dataset/3) |
| 2         | 2    | `Top-Confidence` | [cam2_confbias_topconf96/dataset/3](https://universe.roboflow.com/sbad-dvvax/cam2_confbias_topconf96/dataset/3)    |
| 2         | 1    | `Least-Confidence` | [cam2_confbias_leastconf96/dataset/5](https://universe.roboflow.com/sbad-dvvax/cam2_confbias_leastconf96/dataset/5) |
| 3         | 5    | `Top-Confidence` | [cam3_confbias_topconf96/dataset/2](https://universe.roboflow.com/sbad-dvvax/cam3_confbias_topconf96/dataset/2)    |
| 3         | 5    | `Least-Confidence` | [cam3_confbias_leastconf96/dataset/3](https://universe.roboflow.com/sbad-dvvax/cam3_confbias_leastconf96/dataset/3) |



## Prerequisites <a name="prerequisites"></a>
Before beginning the installation process, ensure you have:
- A Linux 20.04 system.
- An active [wandb](https://wandb.ai/) account for experiment tracking.
- Conda or virtualenv prepared on your machine.
- Videos dataset split according to the Dataset section of at least 5000 samples per image


## Installation <a name="installation"></a>

The code was developed under Linux 20.04. 

### Setup your virtual environment 

We recommend working in a virtualenv or conda environment.

```bash
conda create -y --name CSBAD python pip
conda activate CSBAD
```
### Requirements

To reproduce the results, you need to install the requirements of the YOLOv8 framework AND:

```bash
cd ..
pip install -r requirements.txt
```
### Configure wandb

We use wandb to log all experimentations. You can either use your own account, or create a team. Either way, you will need to login and setup an entity to push logs and model versions to.

1. Create a [wandb](https://wandb.ai/) entity
2. Setup wandb to send logs :

```bash
wandb login
```


## Datasets <a name="datasets"></a>

Required dataset structure :

![Dataset Structure](images/traill22_dataset_structure.svg)

Modifications to this structure are permissible but require appropriate configuration adjustments.

Ensure your dataset adheres to the following structure:

Dataset
├── cam{1}
│   ├── week{1}
│   │   └── bank
│   │   │   ├── images
│   │   │   └── labels_${STUDENT-MODEL}_w_conf 
        |   └── labels_${TEACHER-MODEL} 
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

## 3. Getting Started <a name="getting-started"></a>

### Generation of the pseudo labels (populate bank)

To generate the pseudo labels, execute the following command:

```bash
python generate_pseudo_labels.py --folder "YOURPATH/WALT"

OPTIONS:
--extension "jpg" or "png"
--model-name "yolov8n, yolov8s, yolov8m, yolov8l, yolov8x6" or any custom model
--output-conf # Produce labels with confidence, useful for confidence_based strategies

```
*Note:* The 'bank' folder must contain an 'images' folder with all the images. If you are on Windows, only use `"` and **not** `'`.

### Conduct an Experiment

Before conducting an experiment, ensure your wandb entity and the project are correctly set up in the `experiments/model/yolov8.yaml` Hydra config file.

To conduct an experiment, follow these steps:

1. Populate a `val` folder based on the data contained in the bank folder.
2. Populate a `train` folder based on a `strategy` applied on the data contained in the bank folder. The `strategy` ensures that no images are duplicated between the train and the val sets.
3. Launch a training on a `yolov8n` model based on the previously generated sets. The scripts automatically launch everything to wandb.

You can launch an experiment by executing the main script:

```bash
python train.py
```

In case of debugging, you can add `HYDRA_FULL_ERROR=1` as an environment variable to see the traceback more clearly.

```bash
HYDRA_FULL_ERROR=1 python train.py
```

#### Modify the configs to change the experiments
Hydra is a configuration management tool that retrieves the information included in the `*.yaml` files to facilitate the deployment of the application.

To modify an experiment you can modify the configuration file `experiments/experiment.yaml`. **At your first use, you will have to modify the paths to the dataset and your wandb username.**

You need to modify 
1. experiments/model/yolov8.yaml (WANDB entity)
2. experiments/experiment.yaml (Insert your data folder)
The logs and outputs of the runs are stored in the `output` folder.

*remark*: if you are using Windows, do not forget to adapt your paths by using `/` instead of **not** `\` or `//`.

>**IMPORTANT !**
>
> We use a specific run naming format to track the experiments in wandb and run testing. We do that using the name attribute in the dataset config file. Look at `experiments/dataset/WALT.yaml` for an example.
> NAMING CONVENTION: {Source_domain}_{Target_domain}_{Student}_{Teacher}_{Strategy}>_{Active-Learning-Setting}_{Total_Samples}{Iteration_Level}_{Epochs}_> >_{Validation_Set} Finer-granularity can be expected as {Source_domain} = {dataset}->{domain}-{period}
 

#### Configure a particular training 


python train.py --multirun n_samples=300 cam=1o2o3 week=1o1o5 strategy=thresh-top-confidence student=yolov8n teacher=yolov8m

This will train by sampling 100 samples from camera 1,2,3 using thresh-top-confindence based on the student_yolov8n confidence. The teacher model generates afterwards a yolov8x6 teacher. 



## 4. Testing <a name="testing"></a>

You can use the download to get all the models of a specific project from wandb. Then you use the inference tool to test the models on the dataset. Finally use the inference_coco tool to generate the same testing metrics for the student and teacher models. All the testing results are concatenated in a single file.

We have created a `test.py` file that executes all of these steps in a single command.

```
python test.py --run-prefix WALT --entity YourEntity --project WALT --template testing/templates/WALT.yaml --dataset_path "YOURPATH/WALT/"
```

Flags :
- `--entity` : wandb project team
- `--project` : wandb project name
- `--run-prefix` : project name used as prefix in the runs names (sometimes it's different than the wandb project name, like in the case of "study")
- `--template` : template file for data.yaml used to specify test "sub" datasets
- `--dataset_path` : parent path containing all dataset folders (camX in case of WALT) 
- `--query_filter` : you can choose to download and test only specific models by filtering through characters or words in the run names.
- `--wandb-download` : you can set this to false, if you would like to run all testing pipeline without the download script


Results will be in `testdir/project` where you'll find : 
1. `wandb` folder containing downloaded weights. 
2. `inference_results.csv` file containing inference results
3. `plots` folder containing generated plots for each metric in the inference results (To modify plots look in `testing/plot.py`)

You can also run each script/tool individually :

1. Download models :
```
python testing/download.py -e YourEntity -p WALT -f ./testdir/WALT/wandb -lf -d
```

2. Test downloaded models on the test set :
```
python3 ./testing/inference.py -w ./testing/WALT/wandb -d "YOURPATH/WALT/" -p WALT -y testing/templates/WALT.yaml -f test -c ./testdir/WALT/inference_results.csv
```

3. Test pretrained Student and Teacher models on dataset :
```
python3 testing/inference_coco.py --model yolov8n --csv_path ./testdir/WALT/inference_results.csv --dataset "s05c016->"YOURPATH/WALT/s05c016/" --dataset--data-template testing/templates/Ai-city.yaml --folder test
```

4. Plot graphs :
```
python testing/plot.py --csv_path ./testdir/WALT/inference_results.csv --save_path ./testdir/Ai-city/plots
```

Use the `--help` flag for more information on the usage of each script.

# Clustering Reproducibility 
To obtain the clustering, you have to train one model per camera and then cross-evaluate this model on each test sets. Namely: 

python train.py n_samples=256 cam=X week=Y strategy=thresh-top-confidence, where (X,Y) is the following pairs (1,1);(2,1);(3,5,);(4,2);(5,3);(6,4);(7,4);(8,3);(9,1). This will use the default settings as the student=yolov8n, teacher=yolov8x6 and epochs=100. 

You can then use the testing module and rearrange the resulting csv. For convenience, we already provided the ``cross_performance.csv" and you can find out the clusters by running 
```
python python cluster_cameras.py
```









