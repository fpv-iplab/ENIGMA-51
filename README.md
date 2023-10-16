# The ENIGMA-51 Dataset
This is the official github repository related to the ENIGMA-51 Dataset.

<div align="center">
  <img src="Images/enigma_16_videos_compressed.gif"/>
</div>

ENIGMA-51 is a new egocentric dataset composed of 51 videos acquired in an industrial laboratory. The subjects interact with industrial tools such as an electric screwdriver and pliers, as well as with electronic instruments such as a power supply and an oscilloscope while executing the steps to complete the procedure. ENIGMA-51 has been annotated with a rich set of annotations which allows to study large variety of tasks, especially tasks related to human-object interactions.

You can download the ENIGMA-51 dataset and its annotations from the [project web page](https://iplab.dmi.unict.it/ENIGMA-51/).


## Citing the ENIGMA-51 Dataset
If you find our work useful in your research, please use the following BibTeX entry for citation.
```
@misc{ragusa2023enigma51,
    title={ENIGMA-51: Towards a Fine-Grained Understanding of Human-Object Interactions in Industrial Scenarios}, 
    author={Francesco Ragusa and Rosario Leonardi and Michele Mazzamuto and Claudia Bonanno and Rosario Scavo and Antonino Furnari and Giovanni Maria Farinella},
    year={2023},
    eprint={2309.14809},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```

## Table of Contents

* [Model Zoo and Baselines](#model-zoo-and-baselines)
* [Visualization Script for Object and Hand Keypoints with Mask Annotations](#visualization-script-for-object-and-hand-keypoints-with-mask-annotations)


## Model Zoo and Baselines:

* [Untrimmed Action Detection](#untrimmed-action-detection)
* [Egocentric Human-Object Interaction Detection](#egocentric-human-object-interaction-detection)
* [Short-Term Object Interaction Anticipation](#short-term-object-interaction-anticipation)
* [NLU of Intents and Entities](#nlu-of-intents-and-entities)

## Untrimmed Action Detection

### Introduction
The instructions below will guide you on replicating the baseline for the Untrimmed Action Detection task or training your own model.
The baseline is based on [ActionFormer](https://arxiv.org/abs/2202.07925), refer to the [official repository](https://github.com/happyharrycn/actionformer_release) for more details.

### Download Features, Annotations, and other needed files
* Download *enigma_UAD.tar.gz* from [this link](https://iplab.dmi.unict.it/fpv/).
* The file includes features, action annotations in JSON format, the custom dataset file (.py), and 3 different config files for each task variant (ht_hr, fc_hd, ht_hr_fc_hd).

**Details**: The features are extracted from a two-stream network pretrained on ActivityNet. Each video chunk is set to a size of 6, and there is no overlapping between adjacent chunks. With a video frame rate of 30, we get 5 chunks per second. For appearance features, we extract data from the Flatten 673 layer of ResNet-200 from the central frame of each chunk. Motion features are extracted from the global pool layer of BN-Inception from optical flow fields computed from the 6 consecutive frames within each chunk. Motion and appearance features are then concatenated.

**Needed steps**
* Features and annotations should be placed under *./data/enigma*
* Config files should be placed under *./configs*
* The custom dataset file should be placed under *./libs/datasets*
* In the `libs/datasets/__init__.py` file, include the import of `enigma` (the dataset name in the custom dataset file set within `@register_dataset()`).
* In the `eval.py` file, replace all the instances of **"val_split"** with **"test_split"**.

The folder structure should look like this: 
```
This folder
│   README.md
│   ...  
|
└───configs/
│    └───enigma_ht_hr_fc_hd.json
│    └───enigma_ht_hr.json
│    └───enigma_fc_hd.json
│    └───...
|
└───data/
│    enigma/
│    │   └───annotations
│    │        └───enigma_ht_hr_fc_hd.json
│    │        └───enigma_ht_hr.json
│    │        └───enigma_fc_hd.json
│    │   └───features   
│    └───...
|
└───libs/
|     └───datasets
|     |      └───enigma.py
│     |      └───...
│     └───...
│   ...
```

### Training and Evaluation
* Choose the config file for training ActionFormer on ENIGMA-51.
* Train the ActionFormer network. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.
```shell
python ./train.py ./configs/enigma_ht_hr.yaml --output reproduce
```
* Save the predictions of the trained model by running this script.
```shell
python ./eval.py ./configs/enigma_ht_hr.yaml ./ckpt/enigma_ht_hr_reproduce --saveonly
```
* To evaluate the trained model, you should run the [mp_mAP.py](UAD/mp_mAP/mp_mAP.py) file, specifying the path to the prediction file, and the path to the testing ground truth file. For more details, please refer to the [mp_mAP.py](UAD/mp_mAP/mp_mAP.py).py file.

### Evaluating on Our Pre-trained Model

We also provide the pre-trained models for the 3 different variants of the task (ht_hr, fc_hd, ht_hr_fc_hd). The models with the relative configs can be downloaded from [this link](https://iplab.dmi.unict.it/fpv/). To evaluate the pre-trained model, please follow the steps listed below.

* Move the config files to the config folder or specify the right path in the script below.
* Create a folder *./pretrained*, then a folder for each task variant and move the weight file under them.
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───pretrained/
│    └───enigma/
│    |   └───ht_hr
│    |    └───...    
│    |   └───fc_hd
│    |    └───... 
│    |   └───ht_hr_fc_hd
│    |    └───...
│    |    
│    └───...
|
└───libs
│
│   ...
```
* Save the predictions of the trained model by running this script.
```shell
python ./eval.py ./configs/enigma_ht_hr.yaml ./pretrained/enigma/ht_hr --saveonly
```
* To evaluate the trained model, you should run the [mp_mAP.py](UAD/mp_mAP/mp_mAP.py) file, specifying the path to the prediction file, and the path to the testing ground truth file. For more details, please refer to the [mp_mAP.py](UAD/mp_mAP/mp_mAP.py).py file.


## Egocentric Human-Object Interaction Detection

This section describes the Egocentric Human-Object Interaction Detection task and our proposed approach.

## Short-Term Object Interaction Anticipation

#### StillFast model
We provided the best model trained on the Training Set of the ENIGMA-51 Dataset.
| architecture | model | config |
| ------------- | ------------- | -------------| 
| StillFast | [link](https://iplab.dmi.unict.it/sharing/ENIGMA-51/StillFast_ENIGMA-51_epoch_19.ckpt) | configs/STA_config.yaml |

Please, refer to the official page of [StillFast](https://github.com/fpv-iplab/stillfast) for additional details.

## NLU of Intents and Entities

This section describes the NLU of Intents and Entities task and our proposed approach.

## Visualization Script for Object and Hand Keypoints with Mask Annotations

This script is designed to visualize object and hand keypoints with mask annotations using preprocessed data files. It utilizes various libraries like OpenCV, NumPy, Matplotlib, and Torch to load and display the data. The script assumes that you have already downoloaded the required JSON and npy files, as it loads them to visualize the annotations.
Hands keypoint            |  Object and hand mask
:-------------------------:|:-------------------------:
![](./Images/keypoint.png)  |  ![](./Images/masks.png)

### Prerequisites🛠️

Before running the script, ensure you have the following:

- Python 3.x installed. 🐍
- Conda package manager installed (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Setup🔧

1. Create a new Conda environment and install the required packages:
```bash
conda create -n your_env_name python=3.x
conda activate your_env_name
conda install matplotlib numpy torch torchvision opencv
```
Replace your_env_name with the desired environment name, and replace 3.x with the desired Python version (e.g., 3.7, 3.8, or 3.9).💻

### Usage 🚀

Activate the Conda environment:
```bash
conda activate your_env_name
```

Make sure to replace your_env_name with the environment name you created.
The script will load the JSON and npy files and visualize the annotations on a sample image.📑🔍

### Outputs 🖼️
The script will display a plot containing the following:
Blue circles representing hand keypoints on a resized sample image from the dataset. 👉🔵
Colored polygons representing object masks with class-specific colors. 🎨🔴🟢🟡🟣


### Note 📝
The script uses random selection to display annotations for a random key from the dataset. If you want to visualize annotations for a specific key, modify the "random_key" variable in the script to the desired key. 🎲
The class_colors dictionary can be modified to map class IDs to your preferred colors. 🎨🔤
Feel free to modify the script as per your requirements, such as customizing colors, filtering keypoints, or adjusting image sizes. Happy visualizing! 🎉🔍

