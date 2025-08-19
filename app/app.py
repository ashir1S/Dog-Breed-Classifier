### 1. Imports and class names setup ###
import gradio as gr
import os
import torch
import pickle
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Dict
from timeit import default_timer as timer

from model import create_resnet50_model # Import the model creation function

# Setup class names - Load from the saved pickle file
classes_save_path = 'classes.pkl' # Assuming classes.pkl is in the same directory as app.py (will copy later)
loaded_classes = []
try:
    with open(classes_save_path, 'rb') as f:
        loaded_classes = pickle.load(f)
    class_names = loaded_classes
    print(f"Classes list loaded successfully from {classes_save_path}")
except FileNotFoundError:
    print(f"Error: Classes file not found at {classes_save_path}. Please ensure the file exists.")
    # Provide a fallback or raise an error if classes are essential
    class_names = [] # Fallback to an empty list
except Exception as e:
    print(f"An error occurred while loading the classes list: {e}")
    class_names = [] # Fallback to an empty list


### 2. Model and transforms preparation ###

# Create ResNet50 model
# Use the number of loaded classes, or a default if loading failed
num_classes = len(class_names) if class_names else 157 # Use the actual number of classes
loaded_model = create_resnet50_model(num_classes=num_classes)


# Load saved weights
# Assuming the model file is in the same directory as app.py (will copy later)
model_filename = "resnet50_dog_breed_classifier_acc_90.7095.pth" # **Update this filename to match your saved model**
model_save_path = model_filename

# Check if the model file exists before loading
if os.path.exists(model_save_path):
    try:
        loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu"))) # Load to CPU
        print(f"Model state dictionary loaded successfully from {model_save_path}")
    except Exception as e:
        print(f"An error occurred while loading the model state dictionary: {e}")
else:
    print(f"Error: Model file not found at {model_save_path}. Please ensure the file exists.")


# Define transforms (using validation transforms)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    # Ensure img is a PIL Image before applying transforms
    img = val_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    loaded_model.eval() # Use the loaded_model
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        # Ensure the model is on the correct device for inference
        # Assuming the model is on CPU as loaded with map_location
        pred_probs = torch.softmax(loaded_model(img), dim=1).squeeze(0) # Use the loaded_model and remove batch dimension

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    if class_names: # Ensure class_names is not empty
        pred_labels_and_probs = {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))}
    else:
        pred_labels_and_probs = {"Error": 1.0} # Handle case where class names were not loaded

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Dog Breed Classifier 🐶"
description = "A ResNet50 model trained to classify images of dog breeds."
article = "Built using PyTorch and Gradio." # You can update this with a link to your notebook or project if desired.

# Create examples list from "examples/" directory
# This assumes the examples directory is a subdirectory of where app.py is located
example_list = [["examples/" + example] for example in os.listdir("examples")]


# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
if __name__ == "__main__":
    demo.launch()
