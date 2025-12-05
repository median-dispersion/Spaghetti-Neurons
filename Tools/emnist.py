import argparse
import struct
import numpy as np

# =================================================================================================
# Read in the images from the binary EMNIST handwritten digits dataset
# =================================================================================================
def read_images(file_path):

    with open(file_path, "rb") as file:

        magic, image_count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051: raise ValueError(f"Invalid image file magic number: {magic}")
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(image_count, rows*cols)

    return images

# =================================================================================================
# Read in the labels from the binary EMNIST handwritten digits dataset
# =================================================================================================
def read_labels(file_path):

    with open(file_path, "rb") as file:

        magic, labels_count = struct.unpack(">II", file.read(8))
        if magic != 2049: raise ValueError(f"Invalid label file magic number: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    return labels

# =================================================================================================
# Write the images to a binary file that can be used as activations for the input neurons
# =================================================================================================
def write_images(file_path, images):

    with open(file_path, "wb") as file:

        file.write(struct.pack("<Q", len(images)))
        file.write(struct.pack("<Q", len(images[0])))

        for image in images:

            file.write(image.astype(np.float64) / 255.0)

# =================================================================================================
# Write the the labels to binary file that can be used as targets for the output neurons
# =================================================================================================
def write_labels(file_path, labels):

    with open(file_path, "wb") as file:

        file.write(struct.pack("<Q", len(labels)))
        file.write(struct.pack("<Q", 10))

        for label in labels:

            digits = np.zeros(10, dtype=np.float64)
            digits[label] = 1.0
            file.write(digits.tobytes())

# =================================================================================================
# Main
# =================================================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for converting the binary EMNIST handwritten digit dataset files into a format that can used by the neural network. This will increase the filesize significantly!")

    parser.add_argument("--images", type=str, required=True, help="File path to the input EMNIST images binary")
    parser.add_argument("--labels", type=str, required=True, help="File path to the input EMNIST labels binary")
    parser.add_argument("--images-output", type=str, required=False, help="File path to the output images binary")
    parser.add_argument("--labels-output", type=str, required=False, help="File path to the output labels binary")

    arguments = parser.parse_args()

    if arguments.images_output is None: arguments.images_output = arguments.images + ".bin"
    if arguments.labels_output is None: arguments.labels_output = arguments.labels + ".bin"

    images = read_images(arguments.images)
    labels = read_labels(arguments.labels)

    if len(labels) != images.shape[0]:
        raise ValueError("Number of labels and images do not match!")

    write_images(arguments.images_output, images)
    write_labels(arguments.labels_output, labels)