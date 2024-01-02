from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import shutil

def data_augmentation(input_dir, output_dir):
    # List all classes (assuming each subdirectory represents a class)
    classes = os.listdir(input_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Clear existing files in the directory
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # Dictionary to store file paths for each class
    data = {cls: [] for cls in classes}

    # Gather file paths for each class
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        data[cls] = [os.path.join(cls_dir, file) for file in os.listdir(cls_dir)]

    # ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate and save augmented images
    for cls, files in data.items():
        cls_augmented_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_augmented_dir)

        for file in files:
            img = load_img(file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=cls_augmented_dir, save_prefix=cls, save_format='jpg'):
                i += 1
                if i >= 5:  # Generate 5 augmented images for each original image
                    break

# Usage example:
input_directory = "D:/Drone/New folder (2)/asl_alphabet_train"
output_directory = "D:/Drone/New folder (2)/asl_alphabet_train_augmented"

data_augmentation(input_directory, output_directory)
