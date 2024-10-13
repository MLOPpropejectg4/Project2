import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Define the model architecture (same as when you trained it)
model = models.resnet18(pretrained=False)  # Pretrained=False since we're loading our own weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (binary classification)
print(model)
# Load the model's state_dict
path = "/Users/atoukoffikougbanhoun/Desktop/AMMI/MLops_project2/project2/project2/Project2/models/model1.ckpt"
model.load_state_dict(torch.load(path))
model.eval()

# Image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path:list, model):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image).numpy()  # Extract feature vector
    return features.flatten()

import os
import random
import numpy as np
path = "/Users/atoukoffikougbanhoun/Desktop/AMMI/MLops_project2/project2/project2/Project2/data/data2"
import os

def select_image_paths(folder_path = path,num = 100,cl = "dog",types = "Train"):
    """
    Returns a list of full file paths in the given folder.

    Parameters:
    - folder_path (str): The path to the folder.

    Returns:
    - List[str]: A list of full paths of the files in the folder.
    """
    # List all files in the folder
    folder_path = folder_path + f"/{types}/{cl}"
    file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))][:num]
    
    return file_list


print(select_image_paths(cl="dog",types="Test"))
print(len(select_image_paths()))


features_train = np.array([extract_features(image_path=i,model=model) for i in select_image_paths(types="Train")])
features_test = np.array([extract_features(image_path=i,model=model) for i in select_image_paths(types="Test")]) #.reshape(-1,)

#print(features)
#print(features.shape)

from scipy import stats
import numpy as np

def detect_drift(train_features, test_features):
    # Kolmogorov-Smirnov test for distribution difference
    ks_stat, p_value = stats.ks_2samp(train_features.flatten(), test_features.flatten())
    return ks_stat, p_value

ks_stat,p_value = detect_drift(features_train,features_test)

def plot_high_level_feature(cl = "cat",mode1 = "Train",mode2 = "Test" ):

    features_train = np.array([extract_features(image_path=i,model=model) for i in select_image_paths(cl = cl,types=mode1)])
    features_test = np.array([extract_features(image_path=i,model=model) for i in select_image_paths(cl = cl,types=mode2)])
    
    data_frame_train = pd.DataFrame(data=features_train,columns=[f"feature{i}" for i in range(features_train.shape[1])])
    data_frame_test = pd.DataFrame(data=features_test,columns=[f"feature{i}" for i in range(features_test.shape[1])])




    plt.scatter(x=data_frame_train["feature0"],y =data_frame_train["feature1"],color='blue', label=f'reference Data /{cl}')
    plt.scatter(x=data_frame_test["feature0"],y =data_frame_test["feature1"],color='red', label=f'Current Data/{cl}')
    plt.xlabel("most important feature1")
    plt.ylabel("most important feature2")
    #plt.text(x= "d",y = "djd",s="d")
    plt.text(data_frame_train["feature0"].mean()+2,
             data_frame_train["feature1"].mean(),
             f"p_value = {p_value} \n test_ks  = {ks_stat}",
             fontsize = 12,color ='black')
    plt.legend()
    plt.show()

plot_high_level_feature()
plot_high_level_feature(cl = "dog")

