import math
import cv2
import tensorflow as tf
import numpy as np
from scipy.ndimage import center_of_mass


def largest_connected_component(img):
    """
    Returns largest component
    Input:
    img: Image array
    Output:
    img: Modified image array
    """
    img = img.astype('uint8')
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        img, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        blank_image = np.zeros(img.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img = np.zeros(output.shape)
    img.fill(255)
    img[output == max_label] = 0
    return img


def image_transform(img):
    """
    Prepare the image for digit recognition
    Inputs:
    img: Image array of Sudoku
    Output:
    img: Binary Image array of Sudoku
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return img


def model_input_format(img, image_size):
    """
    Converts image to be compatible with tensorflow model
    Inputs:
    img: Image array to be fed to model
    image_size: Dimensions of image to be fed to model
    Output:
    processed_img: Image array in the format accepted by tensorflow model
    """
    processed_img = tf.reshape(
        img, (image_size[0], image_size[1], 1))
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


def trim_borders(img, ratio=0.6):
    """
    Trims borders of images
    Inputs:
    img: Input image to be trimmed
    ratio: Ratio of blank space to filled space
    Outputs:
    img: Image with trimmed borders
    """
    while np.sum(img[0]) <= (1-ratio) * img.shape[1] * 255:
        img = img[1:]
    while np.sum(img[:, -1]) <= (1-ratio) * img.shape[1] * 255:
        img = np.delete(img, -1, 1)
    while np.sum(img[:, 0]) <= (1-ratio) * img.shape[0] * 255:
        img = np.delete(img, 0, 1)
    while np.sum(img[-1]) <= (1-ratio) * img.shape[0] * 255:
        img = img[:-1]
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
    offset = math.floor((height + width)/100)
    output_image = img[round(height/9 * i + offset): round(height/9 * (i + 1) - offset),
                       round(width/9 * j + offset): round(width/9 * (j + 1) - offset)]
    output_image = cv2.bitwise_not(output_image)
    output_image = trim_borders(output_image)
    empty_image = cv2.bitwise_not(output_image)
    empty_image = largest_connected_component(empty_image)
    center_width = output_image.shape[1] // 2
    center_height = output_image.shape[0] // 2
    x_start = center_height // 2
    x_end = center_height // 2 + center_height
    y_start = center_width // 2
    y_end = center_width // 2 + center_width
    center_region = empty_image[x_start:x_end, y_start:y_end]
    if center_region.sum() >= center_width * center_height * 250 - 255:
        return 0
    if output_image.sum() >= image_size[0]**2*250 - image_size[1] * 1 * 255:
        return 0
    output_image = np.array(output_image)
    output_image = cv2.bitwise_not(output_image)
    output_image = shift_image(output_image)
    output_image = cv2.bitwise_not(output_image)
    output_image = cv2.resize(output_image, image_size)
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
    img = image_transform(img)
    image_array = np.zeros((81, 28, 28, 1))
    index_array = []
    count = 0
    try:
        if height > 10 and width > 10:
            result_array = np.zeros([9, 9])
            result_max = np.zeros([9, 9])
            for i in range(9):
                for j in range(9):
                    temp_image = crop_image(
                        img, i, j, height, width, image_size)
                    if isinstance(temp_image, int):
                        result_array[i][j] = 0
                        result_max[i][j] = 0
                    else:
                        temp_image = model_input_format(temp_image, image_size)
                        image_array[count] = temp_image
                        index_array.append((i, j))
                        count += 1

            predictions = model.predict(image_array)
            count = 0
            for (i, j) in index_array:
                result_array[i][j] = int(np.argmax(predictions[count]))
                result_max[i][j] = np.max(predictions[count])
                count += 1
            if result_array.sum() > 0:
                if np.true_divide(result_max.sum(), np.count_nonzero(result_max)) > 0.95:
                    return result_array
            else:
                pass
    except IndexError:
        pass
    except TypeError:
        pass
