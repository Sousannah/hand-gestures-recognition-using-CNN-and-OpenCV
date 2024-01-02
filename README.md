# Gesture Recognition with CNN and OpenCV

This project aims to recognize hand gestures using Convolutional Neural Networks (CNN) and OpenCV. It involves data augmentation, CNN model training, and integration with OpenCV for real-time gesture recognition.

## Overview

This repository contains scripts and models for:
- Data augmentation using Keras' ImageDataGenerator.
- CNN model creation and training for hand gesture recognition.
- OpenCV integration to detect and classify gestures in real-time.

## Models Available

### CNN_model01.h5

This model, trained with augmented data, initially achieved an accuracy of around 95.67% on the test data. However, it exhibited signs of overfitting, affecting its generalization.

### CNN_model02.h5 - Overfitting Mitigated

The second model, `CNN_model02.h5`, was developed to address overfitting concerns observed in the initial model. Through strategic adjustments, including added dropout layers and regularization techniques, this model aims to mitigate overfitting issues while maintaining or improving overall accuracy.

## Setup and Usage

### Data Augmentation
- `data_augmentation.py`: Script to perform data augmentation on the input dataset.
- Usage example:
  ```bash
  python data_augmentation.py input_directory output_directory
  ```

### CNN Model Training
- `model_training.py`: Script to build, train, and evaluate the CNN model.
- Usage example:
  ```bash
  python model_training.py
  ```

### Gesture Detection with OpenCV
- `gesture_detection.py`: Code utilizing OpenCV and cvzone for real-time gesture recognition.
- Usage:
  - Ensure all dependencies are installed (`requirements.txt`).
  - Run the script to access the webcam and perform gesture recognition.

## Repository Structure

- `code/`: Contains all code scripts.
- `assets/`: Stores additional resources like images, labels, and models.
- `documentation/`: Detailed explanation and analysis of code functionality.

## Dependencies

- Python 3.x
- Libraries: Keras, OpenCV, cvzone, matplotlib, pandas, scikit-learn, seaborn

## Model Improvement (Second Model)

The `CNN_model02.h5` script addresses overfitting by incorporating dropout layers and regularization techniques. It includes usage instructions and documentation within the codebase.

## Performing on Real-Life Video

### Screenshots

![Gesture Detection](path_to_screenshot.png)
*Caption: Real-time gesture detection using OpenCV.*

## Dataset

The dataset used for training and testing the models can be found on [Kaggle]([link_to_dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)).
We only used the 'A' 'B' 'C' 'F' 'K' 'Y'
We name them as the following: "Down", "Up", "Right", "Back", "Front","Left". As these will be a distinct moves for the drone on AirSim

## Gesture Control for AirSim Drone
`AirSim_Drone_Control.py`
The recognized gestures correspond to distinct movements for the AirSim drone:

Down: Move the drone downwards.
Up: Move the drone upwards.
Right: Move the drone to the right.
Back: Move the drone backwards.
Front: Move the drone forwards.
Left: Move the drone to the left.
These gestures trigger specific actions for the drone when detected.
## Contributing

Feel free to contribute by forking the repository and submitting pull requests. Bug fixes, enhancements, and new features are welcome!

## License

This project is licensed under the [MIT License](LICENSE).
```

Please replace `path_to_screenshot.png` with the actual path to your screenshot showcasing the model's performance in real-time gesture detection. Additionally, replace `link_to_dataset` with the appropriate link to your dataset on Kaggle or any other platform.
