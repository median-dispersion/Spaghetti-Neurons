import argparse
import struct
import json

parser = argparse.ArgumentParser(description="A script for converting a spaghetti neurons binary file (.sn) to JSON")

parser.add_argument("--input", type=str, required=True, help="File path to the input binary file")
parser.add_argument("--output", type=str, required=False, help="Output neural network JSON file")

arguments = parser.parse_args()

if arguments.output is None: arguments.output = arguments.input + ".json"

network = {
    "layers": []
}

with open(arguments.input, "rb") as file:

    for l in range(struct.unpack("<Q", file.read(8))[0]):

        layer = {
            "neurons": []
        }

        for n in range(struct.unpack("<Q", file.read(8))[0]):

            neuron = {
                "bias": struct.unpack("<d", file.read(8))[0],
                "connections": []
            }

            for c in range(struct.unpack("<Q", file.read(8))[0]):

                connection = {
                    "target": struct.unpack("<Q", file.read(8))[0],
                    "weight": struct.unpack("<d", file.read(8))[0]
                }

                neuron["connections"].append(connection)

            layer["neurons"].append(neuron)

        network["layers"].append(layer)

with open(arguments.output, "w") as file:

    json.dump(network, file, indent=4)