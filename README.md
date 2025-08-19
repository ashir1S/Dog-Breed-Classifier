Dog Breed Classifier
This project is a dog breed classifier built using PyTorch, which has been deployed as a Gradio web application on Hugging Face Spaces. The model is trained on a custom dataset to identify different dog breeds from images.

About the Project
The core of this project is a machine learning model that takes an image of a dog as input and outputs the predicted breed. The project uses a deep learning approach with a pre-trained convolutional neural network (CNN) model that is fine-tuned on a custom dataset.

Dataset
The model was trained on the "Dog Breeds Image Dataset" sourced from Kaggle. This dataset contains a wide variety of dog breed images, which were split into training and validation sets (80% for training, 20% for validation) to ensure the model's accuracy and prevent overfitting.

Dataset Source: darshanthakare/dog-breeds-image-dataset (Kaggle)

Number of Classes: 157 dog breeds

Training Samples: 13,934 images

Validation Samples: 3,552 images

Model Architecture
The project leverages transfer learning, using a pre-trained model from the torchvision.models library. The model's final classification layer was replaced and fine-tuned to classify the 157 specific dog breeds in the dataset.

Framework: PyTorch

Core Library: torchvision

Transforms: The images were preprocessed with standard transformations, including resizing, cropping, and normalization, to prepare them for the model.

Deployment on Hugging Face Spaces
The trained model has been deployed as an interactive web demo using the Gradio library. You can access the live application to test the classifier with your own dog images.

Hugging Face Space: https://huggingface.co/spaces/Ashirwad12/Dog_Breed_Classifier

Gradio Integration: The Gradio library simplifies the process of creating a user-friendly interface for the model, allowing anyone to upload an image and get an instant prediction. The notebook contains an IFrame to embed the Gradio demo.

How to Run Locally
If you wish to run this project locally, you will need to set up your environment and follow these steps:

Clone the repository:

git clone https://github.com/Ashirwad12/Dog-Breed-Classifier.git
cd Dog-Breed-Classifier

Install dependencies:
The required libraries are listed in a requirements file (you may need to create one based on the notebook's imports).

pip install -r requirements.txt

Download the dataset from Kaggle:
You will need to set up your Kaggle API credentials. The notebook uses google.colab.userdata to retrieve a secret API key. You will need to configure your Kaggle credentials locally, usually by placing kaggle.json in a .kaggle directory in your home folder.

Run the Jupyter Notebook:
Execute the Dog_Breed_Classification.ipynb notebook cell by cell to train the model, or load the pre-trained model if available. The notebook contains all the code for data splitting, transformation, and model training.

Contributing
Feel free to open an issue or submit a pull request if you find a bug or have an idea for an improvement.

License
This project is licensed under the MIT License.
