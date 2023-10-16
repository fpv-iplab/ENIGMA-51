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

This section describes the Untrimmed Action Detection task and our proposed approach.

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

