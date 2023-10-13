


## Model Zoo and Baselines:
### 1) Untrimmed Action Detection
### 2) Egocentric Human-Object Interaction Detection

### 3) Short-Term Object Interaction Anticipation

#### StillFast model
We provided the best model trained on the Training Set of the ENIGMA-51 Dataset.
| architecture | model | config |
| ------------- | ------------- | -------------| 
| StillFast | [link]() | configs/STA/STA_config.yaml |

Please, refer to the official page of [StillFast](https://github.com/fpv-iplab/stillfast) for additional details.
### 4) NLU of Intents and Entities
.
.
.









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

