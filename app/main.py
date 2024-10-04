import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import get_classes


###########################################

from fastapi import Depends, HTTPException, status, UploadFile, File
from fastapi.security import  OAuth2PasswordRequestForm

from typing import Annotated

from app.app import app
from utils.auth import authenticate_user, oauth2_scheme



##################################################################################


# Define the model architecture (same as when you trained it)
model = models.resnet18(pretrained=False)  # Pretrained=False since we're loading our own weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (binary classification)

# Load the model's state_dict
path = "/Users/atoukoffikougbanhoun/Desktop/AMMI/MLops_project0/group4/project2/trained_model/model1.ckpt"
model.load_state_dict(torch.load(path))

# Set the model to evaluation mode (important for inference)
model.eval()

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Device configuration (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Classes
classes = get_classes(path="/Users/atoukoffikougbanhoun/Desktop/AMMI/MLops_project0/group4/project2/data/data1/Train")

# Function to undo normalization and convert the tensor to a NumPy array for visualization
def imshow_tensor(img_tensor):
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))  # Convert to HWC format
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # Undo normalization
    img = np.clip(img, 0, 1)  # Clip the values to be between 0 and 1 for display
    return img






############################################################################



@app.get("/")
async def root():
    return {"message": "Healthy"}

@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"Authenticated_user": user["username"], "Password": user["password"]}


@app.post("/predict/")
async def upload_file(
      form_data: Annotated[OAuth2PasswordRequestForm, Depends(oauth2_scheme)], file: UploadFile = File(...)
):
    

    try:
        # Read the uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Apply the transformations
        image = data_transforms(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Perform inference
        with torch.no_grad():  # No need for gradients during inference
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]

        # Convert the input tensor back to an image and prepare it for visualization
        visualized_image = imshow_tensor(image[0])

        # Optionally, save or display the image (you could return this in the API response)
        plt.imshow(visualized_image)
        plt.title(f'Predicted: {predicted_class}')
        plt.axis('off')
        plt.show()

        # Return the prediction
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

print("Server is running correctly")

