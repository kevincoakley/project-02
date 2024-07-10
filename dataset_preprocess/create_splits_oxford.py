import argparse, os, random


def create_splits(root_path):

    train_path = root_path + "/train"
    val_path = root_path + "/val"
    test_path = root_path + "/test"

    classes = [
        f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))
    ]

    # Create the val_path directory if it doesn't exist
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    for data_class in classes:
        train_class_path = train_path + "/" + data_class
        val_class_path = val_path + "/" + data_class
        test_class_path = test_path + "/" + data_class

        # Create the new class directory if it doesn't exist
        if not os.path.exists(val_class_path):
            os.makedirs(val_class_path)

        # Create the new class directory if it doesn't exist
        if not os.path.exists(test_class_path):
            os.makedirs(test_class_path)

        # Get the images in the class directory
        images = []
        for img in os.listdir(train_class_path):
            images.append(train_class_path + "/" + img)

        # Shuffle the images
        random.shuffle(images)
        
        train_len = 10
        val_len = 10
        test_len = len(images) - 20        

        # Split the images
        train_images = images[:train_len]
        images = images[train_len:]
        val_images = images[:val_len]
        images = images[val_len:]
        test_images = images[:test_len]

        # Move the images to the validation folder
        for img in val_images:
            os.rename(img, img.replace(train_path, val_path))
        
        for img in test_images:
            os.rename(img, img.replace(train_path, test_path))


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

    create_splits(args.path)
