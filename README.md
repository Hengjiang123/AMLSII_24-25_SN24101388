# README
This repo contain a complete pipeline of training a EfficientNet model with SIIM melanoma 384x384 jpg and meta data.

This pipeline can Automatically detect the environment and choose correct file path. But you need to set it correctly first.

To run the project:

1. CPU:
Set the environment as description in the folder env/requirements.txt
Prepare the dataset (it shoud be with the code in github repo) and run main.py.

2. GPU in Kaggle
Switch the file path to Kaggle(at top of the main.py, evaluate.py and train.py)
Remove the Dataset folder, and zip this folder as a Dataset in Kaggle.
https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-384x384 <--- Add this Dataset as well to the notabook

!pip install efficientnet_pytorch   <---   install EfficientNet library first
Also set using GPU acceleration and run the notebook. 
