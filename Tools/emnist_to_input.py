import argparse
from pathlib import Path
import struct
import numpy

# =================================================================================================
# Convert the images from the EMNIST handwritten digits dataset binary
# =================================================================================================
def convert_images(file_path):

    with open(file_path, "rb") as file:

        magic, count, height, width = struct.unpack(">IIII", file.read(16))
        size = width * height

        if magic != 2051: raise ValueError("Invalid magic number for the images file!")

        data = numpy.frombuffer(file.read(), dtype=numpy.uint8)
        data = data.astype(numpy.float64) / 255.0
        data = data.reshape(count, height, width)
        data = data.transpose(0, 2, 1)
        data = data.flatten()

    return count, size, data

# =================================================================================================
# Convert the labels from the EMNIST handwritten digits dataset binary
# =================================================================================================
def convert_labels(file_path):

    with open(file_path, "rb") as file:

        magic, count = struct.unpack(">II", file.read(8))
        size = 10

        if magic != 2049: raise ValueError("Invalid magic number for the labels file!")

        data = numpy.zeros(count * size, dtype=numpy.float64)
        buffer = numpy.frombuffer(file.read(), dtype=numpy.uint8)

        for index in range(count):

            offset = buffer[index] + 10 * index
            data[offset] = 1.0

    return count, size, data

# =================================================================================================
# Write the converted images data to a binary file
# =================================================================================================
def write_images(file_path, images_count, images_size, images_data):

    with open(file_path, "wb") as file:

        file.write(struct.pack("<Q", images_count))
        file.write(struct.pack("<Q", images_size))
        file.write(images_data)

# =================================================================================================
# Write the converted labels data to a binary file
# =================================================================================================
def write_labels(file_path, labels_count, labels_size, labels_data):

    with open(file_path, "wb") as file:

        file.write(struct.pack("<Q", labels_count))
        file.write(struct.pack("<Q", labels_size))
        file.write(labels_data)

# =================================================================================================
# Main
# =================================================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for converting the EMNIST handwritten digit dataset binary files into a format that can be used by the neural network as an input. This will increase the file size significantly!")

    parser.add_argument("--images", type=str, required=True, help="File path to the input EMNIST images binary")
    parser.add_argument("--labels", type=str, required=True, help="File path to the input EMNIST labels binary")
    parser.add_argument("--images-output", type=str, required=False, help="File path to the output images binary")
    parser.add_argument("--labels-output", type=str, required=False, help="File path to the output labels binary")

    arguments = parser.parse_args()

    arguments.images = Path(arguments.images)
    arguments.labels = Path(arguments.labels)

    if arguments.images_output is None: arguments.images_output = f"{arguments.images.stem}_images.bin"
    if arguments.labels_output is None: arguments.labels_output = f"{arguments.labels.stem}_labels.bin"

    arguments.images_output = Path(arguments.images_output)
    arguments.labels_output = Path(arguments.labels_output)

    images_count, images_size, images_data = convert_images(arguments.images)
    labels_count, labels_size, labels_data = convert_labels(arguments.labels)

    if images_count != labels_count: raise ValueError("Number of images and labels do not match!")

    write_images(arguments.images_output, images_count, images_size, images_data)
    write_labels(arguments.labels_output, labels_count, labels_size, labels_data)