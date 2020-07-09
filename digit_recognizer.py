import math
import cv2
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from scipy.ndimage import center_of_mass

# model = tf.keras.models.load_model('digit-recognizer.h5')


def model_input_format(img, image_size):
    """
    Converts image to be compatible with tensorflow model
    Inputs: 
    img: Image array to be fed to model
    image_size: Dimensions of image to be fed to model
    Output:
    processed_img: Image array in the format accepted by tensorflow model
    """
    processed_img = tf.expand_dims(tf.reshape(
        img, (image_size[0], image_size[1], 1)), 0)
    return processed_img


def shift_image(img):
    """
    Shift image towards it's center of mass
    Inputs:
    img: Image array
    Output:
    img: Resultant Image array
    """
    center_y, center_x = center_of_mass(img)
    rows, cols = img.shape
    shift_x = np.round(cols/2.0-center_x).astype(int)
    shift_y = np.round(rows/2.0-center_y).astype(int)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, M, (rows, cols))
    return img


def crop_image(img, i, j, height, width, image_size):
    """
    Crops and processes input image
    Inputs:
    img: Image Array
    i: ith row of sudoku
    j: jth column of sudoku
    height: Height of sudoku image
    width: Width of sudoku image
    image_size: Dimensions of image to be fed to the model
    Output:
    0 if image is empty
    output_image: Image array
    """
    ratio = 0.6
    offset = math.floor((height + width)/180)
    output_image = img[round(height/9 * i + offset): round(height/9 * (i + 1) - offset),
                       round(width/9 * j + offset): round(width/9 * (j + 1) - offset)]
    while np.sum(output_image[0]) >= (1-ratio) * output_image.shape[1] * 255:
        output_image = np.delete(output_image, 0, 0)
    while np.sum(output_image[:, -1]) >= (1-ratio) * output_image.shape[1] * 255:
        output_image = np.delete(output_image, -1, 1)
    while np.sum(output_image[:, 0]) >= (1-ratio) * output_image.shape[0] * 255:
        output_image = np.delete(output_image, 0, 1)
    while np.sum(output_image[-1]) >= (1-ratio) * output_image.shape[0] * 255:
        output_image = np.delete(output_image, -1, 0)

    if output_image.sum() >= image_size[0]**2*255 - image_size[1] * 1 * 255:
        return 0
    center_width = output_image.shape[1] // 2
    center_height = output_image.shape[0] // 2
    x_start = center_height // 2
    x_end = center_height // 2 + center_height
    y_start = center_width // 2
    y_end = center_width // 2 + center_width
    center_region = output_image[x_start:x_end, y_start:y_end]

    if center_region.sum() >= center_width * center_height * 255 - 255:
        return 0
    if output_image.sum() < 5000:
        return 0
    # if output_image.sum() >= 90000 || output_image.sum() == 0:
    #   return 0
    output_image = shift_image(output_image)
    output_image = cv2.bitwise_not(output_image)
    output_image = resize(output_image, image_size)
    return output_image


def scan_sudoku(img, model, image_size=(28, 28)):
    """
    Takes sudoku image as input and recognizes the digits present
    Inputs:
    img: Image array of sudoku
    model: Tensorflow model to be used
    image_size: Dimensions of image to be fed to model
    Output:
    result_array: 2D Array of digits recognized with 0 in place of blank spaces
    """
    height, width = img.shape[0], img.shape[1]
    try:
        if height > 10 and width > 10:
            result_array = np.zeros([9, 9])
            for i in range(9):
                for j in range(9):
                    temp_image = crop_image(
                        img, i, j, height, width, image_size)
                    if temp_image is 0:
                        result_array[i][j] = 0
                    else:
                        predictions = model.predict(
                            model_input_format(temp_image, image_size))
                        result_array[i][j] = np.argmax(predictions)
            return result_array
    except IndexError:
        print("Image Not Valid")
