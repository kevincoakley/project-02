import argparse, os
import cv2
import numpy as np


def calculate_channel_stats(directory):
    # Initialize lists to store channel values
    red_values = []
    green_values = []
    blue_values = []

    # Iterate over all files in the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)

            if (
                file_path.lower().endswith(".jpg")
                or file_path.lower().endswith(".jpeg")
                or file_path.lower().endswith(".png")
            ):
                # Read the image
                image = cv2.imread(file_path)

                if image is not None:  # Check if the image is valid
                    # Split the image into color channels
                    red_channel = image[:, :, 2]
                    green_channel = image[:, :, 1]
                    blue_channel = image[:, :, 0]

                    # Append channel values to the respective lists
                    red_values.extend(red_channel.flatten())
                    green_values.extend(green_channel.flatten())
                    blue_values.extend(blue_channel.flatten())

    # Calculate mean and standard deviation for each channel
    red_mean = round(np.mean(red_values) / 255, 4)
    green_mean = round(np.mean(green_values) / 255, 4)
    blue_mean = round(np.mean(blue_values) / 255, 4)

    red_std = round(np.std(red_values) / 255, 4)
    green_std = round(np.std(green_values) / 255, 4)
    blue_std = round(np.std(blue_values) / 255, 4)

    return red_mean, green_mean, blue_mean, red_std, green_std, blue_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        dest="path",
        help="path to the dataset",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    red_mean, green_mean, blue_mean, red_std, green_std, blue_std = (
        calculate_channel_stats(args.path)
    )
    print(f"RBG Mean: ({red_mean}, {green_mean}, {blue_mean})")
    print(f"RBG Std: ({red_std}, {green_std}, {blue_std})")
