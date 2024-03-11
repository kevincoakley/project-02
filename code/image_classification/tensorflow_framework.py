import tensorflow as tf
import tensorflow_datasets as tfds

import csv, datetime, math, random
import numpy as np
from sklearn.metrics import accuracy_score

import densenet_conv4_tensorflow as densenet_conv4
import densenet_conv5_tensorflow as densenet_conv5
import resnet_conv4_tensorflow as resnet_conv4
import resnet_conv5_tensorflow as resnet_conv5
import simple_conv4_tensorflow as simple_conv4
import simple_conv5_tensorflow as simple_conv5


class Tensorflow:
    def __init__(self):
        self.script_version = "1.0.4"
        self.version = tf.version.VERSION
        self.optimizer = "SGD"
        self.nesterov = False
        self.epochs = 0
        self.lr_scheduler = False
        self.lr_warmup = False
        self.learning_rate = 0.0
        self.save_epoch_logs = False
        self.save_tensorboard_logs = False

    def set_random_seed(self, seed_val):
        """
        ## Configure Tensorflow for fixed seed runs
        """
        major, minor, revision = tf.version.VERSION.split(".")

        if int(major) >= 2 and int(minor) >= 7:
            # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
            # Supported in TF 2.7.0+
            tf.keras.utils.set_random_seed(seed_val)
            print("Setting random seed using tf.keras.utils.set_random_seed()")
        else:
            # for TF < 2.7
            random.seed(seed_val)
            np.random.seed(seed_val)
            tf.random.set_seed(seed_val)
            print("Setting random seeds manually")

    def set_op_determinism(self):
        """
        ## Configure Tensorflow for deterministic operations
        """
        major, minor, revision = tf.version.VERSION.split(".")

        # Configures TensorFlow ops to run deterministically to enable reproducible
        # results with GPUs (Supported in TF 2.8.0+)
        if int(major) >= 2 and int(minor) >= 8:
            tf.config.experimental.enable_op_determinism()
            print("Enabled op determinism")
        else:
            print("Op determinism not supported")
            exit()

    def load_dataset(self, dataset_details, batch_size):
        train_path = dataset_details["train_path"]
        val_path = dataset_details["val_path"]
        test_path = dataset_details["test_path"]
        dataset_shape = dataset_details["dataset_shape"]
        normalization_mean = dataset_details["normalization"]["mean"]
        normalization_std = dataset_details["normalization"]["std"]

        def preprocessing(image, label):
            image = tf.cast(image, tf.float32)
            # Normalize the pixel values ((input[channel] - mean[channel]) / std[channel])
            image = tf.divide(
                image, (255.0, 255.0, 255.0)
            )  # divide by 255 to match pytorch
            image = tf.subtract(image, normalization_mean)
            image = tf.divide(image, normalization_std)
            return image, label

        def augmentation(image, label):
            image = tf.image.resize_with_crop_or_pad(
                image, 
                round(dataset_shape[0] + (dataset_shape[0] * .25)), 
                round(dataset_shape[1] + (dataset_shape[1] * .25))
            )
            image = tf.image.random_crop(image, dataset_shape)
            image = tf.image.random_flip_left_right(image)
            return image, label

        # Get the training and validation datasets from the directory
        train = tf.keras.utils.image_dataset_from_directory(
            train_path,
            shuffle=True,
            image_size=dataset_shape[:2],
            interpolation="nearest",
            batch_size=None,
        )
        val = tf.keras.utils.image_dataset_from_directory(
            val_path,
            shuffle=False,
            image_size=dataset_shape[:2],
            interpolation="nearest",
            batch_size=None,
        )
        test = tf.keras.utils.image_dataset_from_directory(
            test_path,
            shuffle=False,
            image_size=dataset_shape[:2],
            interpolation="nearest",
            batch_size=None,
        )

        print(f"Number of training samples: {train.cardinality()}")
        print(f"Number of validation samples: {val.cardinality()}")
        print(f"Number of test samples: {test.cardinality()}")

        # Batch and prefetch the dataset
        train_dataset = (
            train.map(preprocessing)
            .map(augmentation)
            .shuffle(1000)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            val.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )
        test_dataset = (
            test.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

        return train_dataset, val_dataset, test_dataset

    def load_model(self, model_name, dataset_details):
        num_classes = dataset_details["num_classes"]
        dataset_shape = dataset_details["dataset_shape"]

        model_functions = {
            "DenseNet_k12d40": densenet_conv4.densenet_k12d40,
            "DenseNet_k12d100": densenet_conv4.densenet_k12d100,
            "DenseNet_k24d100": densenet_conv4.densenet_k24d100,
            "DenseNet_bc_k12d100": densenet_conv4.densenet_bc_k12d100,
            "DenseNet_bc_k24d250": densenet_conv4.densenet_bc_k24d250,
            "DenseNet_bc_k40d190": densenet_conv4.densenet_bc_k40d190,
            "DenseNet121": densenet_conv5.densenet121,
            "DenseNet169": densenet_conv5.densenet169,
            "DenseNet201": densenet_conv5.densenet201,
            "DenseNet264": densenet_conv5.densenet264,
            "ResNet20": resnet_conv4.resnet20,
            "ResNet32": resnet_conv4.resnet32,
            "ResNet44": resnet_conv4.resnet44,
            "ResNet56": resnet_conv4.resnet56,
            "ResNet110": resnet_conv4.resnet110,
            "ResNet1202": resnet_conv4.resnet1202,
            "ResNet18": resnet_conv5.resnet18,
            "ResNet34": resnet_conv5.resnet34,
            "ResNet50": resnet_conv5.resnet50,
            "ResNet101": resnet_conv5.resnet101,
            "ResNet152": resnet_conv5.resnet152,
            "Simple4_1": simple_conv4.simple4_1,
            "Simple4_3": simple_conv4.simple4_3,
            "Simple4_5": simple_conv4.simple4_5,
            "Simple4_7": simple_conv4.simple4_7,
            "Simple4_9": simple_conv4.simple4_9,
            "Simple4_11": simple_conv4.simple4_11,
            "Simple4_13": simple_conv4.simple4_13,
            "Simple4_15": simple_conv4.simple4_15,
            "Simple5_1": simple_conv5.simple5_1,
            "Simple5_3": simple_conv5.simple5_3,
            "Simple5_5": simple_conv5.simple5_5,
            "Simple5_7": simple_conv5.simple5_7,
            "Simple5_9": simple_conv5.simple5_9,
            "Simple5_11": simple_conv5.simple5_11,
            "Simple5_13": simple_conv5.simple5_13,
            "Simple5_15": simple_conv5.simple5_15,
        }

        model = model_functions[model_name](
            input_shape=dataset_shape, num_classes=num_classes
        )

        model.build(
            input_shape=(None, dataset_shape[0], dataset_shape[1], dataset_shape[2])
        )

        # Print the model summary
        # model.summary()

        return model

    def train(
        self,
        model,
        train_dataset,
        val_dataset,
        epochs,
        csv_train_log_file=None,
        run_name="",
    ):
        """
        ## Define the learning rate schedule
        """

        def lr_schedule(epoch):
            if self.lr_warmup and epoch < 5:
                return self.learning_rate * 0.1
            elif epoch < math.ceil(self.epochs * 0.5):
                return self.learning_rate
            elif epoch < math.ceil(self.epochs * 0.75):
                return self.learning_rate * 0.1
            else:
                return self.learning_rate * 0.01

        if self.optimizer == "SGD":
            model.compile(
                optimizer=tf.keras.optimizers.experimental.SGD(
                    weight_decay=0.0001,
                    momentum=0.9,
                    learning_rate=lr_schedule(0),
                    nesterov=self.nesterov,
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
        elif self.optimizer == "Adam":
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

        # Define the learning rate scheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # Define csv logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(csv_train_log_file)

        # Define tensorboard callback
        log_dir = (
            "logs/fit/"
            + run_name
            + "-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        # Define callbacks
        callbacks = []

        if self.save_epoch_logs:
            callbacks.append(csv_logger)

        if self.lr_scheduler:
            callbacks.append(lr_scheduler)

        if self.save_tensorboard_logs:
            callbacks.append(tensorboard_callback)

        # Train the model
        model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
        )

        return model

    def evaluate(
        self, model, test_dataset, save_predictions=False, predictions_csv_file=None
    ):
        # Get the predictions
        predictions = model.predict(test_dataset)

        # Get the labels of the validation dataset
        test_dataset = test_dataset.unbatch()
        labels = np.asarray(list(test_dataset.map(lambda x, y: y)))

        # Get the index to the highest probability
        y_true = labels
        y_pred = np.argmax(predictions, axis=1)

        if save_predictions:
            # Add the true values to the first column and the predicted values to the second column
            true_and_pred = np.vstack((y_true, y_pred)).T

            # Add each label predictions to the true and predicted values
            csv_output_array = np.concatenate((true_and_pred, predictions), axis=1)

            # Save the predictions to a csv file
            with open(predictions_csv_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["true_value", "predicted_value"]
                for i in range(predictions.shape[1]):
                    csv_columns.append("label_" + str(i))

                writer.writerow(csv_columns)
                writer.writerows(csv_output_array.tolist())

        # Calucate the validation loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        validation_loss = loss(labels, predictions).numpy()

        # Use sklearn to calculate the validation accuracy
        validation_accuracy = accuracy_score(y_true, y_pred)

        return [validation_loss, validation_accuracy]

    def save(self, model, model_path):
        model.save(model_path)

    def load(self, model_path):
        model = tf.keras.models.load_model(model_path)

        return model
