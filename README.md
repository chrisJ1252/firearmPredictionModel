Weapon Detection Using Transfer Learning with MobileNetV2
This project aims to build an image classification model that can detect different types of weapons in images using transfer learning with the pre-trained MobileNetV2 architecture. The model is trained on a custom dataset of weapon images and fine-tuned for classification tasks.

Table of Contents
Project Overview
Dataset
Model Architecture
Data Augmentation
Training Process
Performance
Usage
Requirements
Acknowledgments
Project Overview
The goal of this project is to detect and classify different weapon types in images using a deep learning approach. We utilize transfer learning, leveraging the MobileNetV2 model pre-trained on the ImageNet dataset, and fine-tune it on our specific dataset of weapon images.

This project includes the following steps:

Data Preparation: Unzipping and organizing the image dataset into training and validation sets.
Data Augmentation: Applying a variety of augmentation techniques to expand the dataset.
Model Development: Fine-tuning the MobileNetV2 model with custom layers for classification.
Training: Training the model using augmented data and monitoring validation performance.
Evaluation: Plotting learning rate vs. loss to find the optimal learning rate, and final model evaluation.
Dataset
The dataset used in this project contains images of different types of weapons. The dataset is organized into two directories:

Train: Images used for training the model.
Validation: Images used for validating model performance.
Each image file is named using a convention that indicates the weapon type, and the dataset contains nine different classes of weapons.

Model Architecture
The model is based on MobileNetV2, which is a lightweight architecture designed for mobile and embedded vision applications. The key steps in the model architecture include:

Base Model: MobileNetV2, pre-trained on the ImageNet dataset, with the top layers removed to allow for customization.
Global Average Pooling: A layer to reduce the dimensions of the feature map from the base model.
Flatten Layer: Flattening the pooled features into a 1D vector.
Dense Layer: A fully connected layer with 256 neurons and ReLU activation.
Dropout Layer: A dropout layer to reduce overfitting with a dropout rate of 0.5.
Output Layer: A softmax layer to predict one of the nine weapon classes.
Data Augmentation
To improve model generalization, the following data augmentation techniques were applied:

Rescaling pixel values between 0 and 1.
Random rotations up to 60 degrees.
Random shifts in width and height by up to 30%.
Random shearing and zooming.
Horizontal flipping of images.
Brightness adjustments and channel shifting.
These augmentations help the model learn robust features and generalize better to unseen images.

Training Process
The model was trained using the Adam optimizer with a starting learning rate of 1.9e-4. The loss function used was sparse categorical crossentropy. The key hyperparameters and callbacks include:

Batch Size: 32
Epochs: 30
Learning Rate Scheduler: Adjusted learning rate dynamically based on the performance.
Early Stopping: Monitors validation accuracy, and stops training if there is no improvement after 10 epochs.
Training was performed on a dataset of 571 images, with 143 images used for validation.

Performance
During training, the model's accuracy improved steadily. Validation accuracy reached over 82.8% by the 30th epoch, indicating that the model is capable of distinguishing between different weapon types effectively.

A learning rate vs. loss plot was generated to visualize and find the most optimal learning rate for further tuning.

Usage
Running the Model
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/weapon-detection.git
cd weapon-detection
Prepare the dataset: Ensure the dataset is organized in train and validation directories, as shown below:

kotlin
Copy code
GunImages/
├── weapon_detection/
│   ├── train/
│   └── val/
Run the training script in a Python environment:

bash
Copy code
python train_weapon_detection.py
Dependencies
Install the necessary Python libraries:

bash
Copy code
pip install tensorflow matplotlib numpy
Requirements
Python 3.7+
TensorFlow 2.x
NumPy
Matplotlib
Google Colab (optional, if running in Colab)
Acknowledgments
The pre-trained MobileNetV2 model was sourced from TensorFlow Keras Applications.
The dataset was manually collected for this project.
Thanks to Google Colab for providing the environment used to develop this model.
