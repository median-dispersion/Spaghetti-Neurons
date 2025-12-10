import argparse
from pathlib import Path
from PIL import Image
import numpy
import struct

# =================================================================================================
# Convert the input image into a format the neural network can use
# =================================================================================================
def convert_image(file_path):

    count = 1
    width = 28
    heigh = 28
    size = width * heigh
    data = Image.open(file_path)
    data = data.convert("L")
    data = data.resize((width, heigh))
    data = numpy.array(data, dtype=numpy.float64) / 255.0
    data = data.flatten()

    return count, size, data

# =================================================================================================
# Convert the input label into a format the neural network can use
# =================================================================================================
def convert_label(label):

    count = 1
    size = 10
    data = numpy.zeros(count * size, dtype=numpy.float64)
    data[label] = 1.0

    return count, size, data

# =================================================================================================
# Write the converted input image to a binary file
# =================================================================================================
def write_image(file_path, image_count, image_size, image_data):

    with open(file_path, "wb") as file:

        file.write(struct.pack("<Q", image_count))
        file.write(struct.pack("<Q", image_size))
        file.write(image_data)

# =================================================================================================
# Write the converted label to a binary file
# =================================================================================================
def write_label(file_path, label_count, label_size, label_data):

    with open(file_path, "wb") as file:

        file.write(struct.pack("<Q", label_count))
        file.write(struct.pack("<Q", label_size))
        file.write(label_data)

# =================================================================================================
# Main
# =================================================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for converting an image of a digit and a label into a binary format that the neural network can use as an input.")

    parser.add_argument("--image", type=str, required=True, help="File path to the input image")
    parser.add_argument("--label", type=int, required=True, help="Label of what digit the input image is depicting (0-9)")
    parser.add_argument("--image-output", type=str, required=False, help="File path of the output image binary")
    parser.add_argument("--label-output", type=str, required=False, help="File path of the output label binary")

    arguments = parser.parse_args()

    if arguments.label < 0 or arguments.label > 9: raise ValueError("Input label is outside the allowed range!")

    arguments.image = Path(arguments.image)

    if arguments.image_output is None: arguments.image_output = f"{arguments.image.stem}_image.bin"
    if arguments.label_output is None: arguments.label_output = f"{arguments.image.stem}_label.bin"

    arguments.image_output = Path(arguments.image_output)
    arguments.label_output = Path(arguments.label_output)

    image_count, image_size, image_data = convert_image(arguments.image)
    label_count, label_size, label_data = convert_label(arguments.label)

    write_image(arguments.image_output, image_count, image_size, image_data)
    write_label(arguments.label_output, label_count, label_size, label_data)