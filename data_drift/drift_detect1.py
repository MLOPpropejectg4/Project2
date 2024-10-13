import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn as nn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Load pretrained ResNet18 model
model = models.resnet18(pretrained=False)  # Pretrained=False since we're loading our own weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (binary classification)
print(model)
# Load the model's state_dict
path_model = "/Users/atoukoffikougbanhoun/Desktop/AMMI/MLops_project2/project2/project2/Project2/models/model1.ckpt"
model.load_state_dict(torch.load(path_model))
model.eval()

# Image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#selector of path
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
# Function to extract features from images
def extract_features(image_path, model):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image).numpy()  # Extract feature vector
    return features.flatten()

# # Example paths (replace these with your actual image paths)
train_image_paths = select_image_paths()
#print(train_image_paths)
test_image_paths = select_image_paths(types="Test")

# Extract features for training and test data
train_features = [extract_features(path, model) for path in train_image_paths]
test_features = [extract_features(path, model) for path in test_image_paths]
#print(train_features)
# Convert to DataFrame
train_df = pd.DataFrame(np.vstack(train_features), columns=[f'feature_{i}' for i in range(len(train_features[0]))])
test_df = pd.DataFrame(np.vstack(test_features), columns=[f'feature_{i}' for i in range(len(test_features[0]))])

# Add a label for the dataset type (train/test)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

# Combine the datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(train_df.head())
# Create a data drift report using Evidently's new API
report = Report(metrics=[DataDriftPreset()])

# Run the report comparing train and test datasets
report.run(reference_data=train_df, current_data=test_df)

# Save the report as an HTML file
report.save_html("data_drift_report.html")
