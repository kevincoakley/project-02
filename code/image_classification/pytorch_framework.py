import torch
import torchvision

import csv, math, random
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime

import densenet_conv4_pytorch as densenet_conv4
import densenet_conv5_pytorch as densenet_conv5
import resnet_conv4_pytorch as resnet_conv4
import resnet_conv5_pytorch as resnet_conv5


class Pytorch:
    def __init__(self):
        self.script_version = "1.0.2"
        self.version = torch.__version__
        self.device = torch.device("cuda:0")
        self.optimizer = "SGD"
        self.nesterov = False
        self.epochs = 0
        self.lr_scheduler = False
        self.lr_warmup = False
        self.learning_rate = 0.0
        self.save_epoch_logs = False
        self.save_tensorboard_logs = False


    def set_random_seed(self, seed_val):
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

    def set_op_determinism(self):
        torch.use_deterministic_algorithms(True)

    def load_dataset(self, dataset_details, batch_size):
        train_path = dataset_details["train_path"]
        val_path = dataset_details["val_path"]
        test_path = dataset_details["test_path"]
        dataset_shape = dataset_details["dataset_shape"]
        normalization_mean = dataset_details["normalization"]["mean"]
        normalization_std = dataset_details["normalization"]["std"]

        normalize = torchvision.transforms.Normalize(
            mean=normalization_mean, std=normalization_std
        )

        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((dataset_shape[0], dataset_shape[1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=False),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

        preprocessing_augument = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((dataset_shape[0], dataset_shape[1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=False),
                torchvision.transforms.Pad((round(dataset_shape[0] * .25), 
                                            round(dataset_shape[1] * .25))),
                torchvision.transforms.RandomCrop((dataset_shape[0], dataset_shape[1])),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

        train = torchvision.datasets.ImageFolder(
            root=train_path, transform=preprocessing_augument
        )
        val = torchvision.datasets.ImageFolder(root=val_path, transform=preprocessing)
        test = torchvision.datasets.ImageFolder(root=test_path, transform=preprocessing)

        train_dataset = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_dataset = torch.utils.data.DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_dataset = torch.utils.data.DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_dataset, val_dataset, test_dataset

    def load_model(self, model_name, dataset_details):
        num_classes = dataset_details["num_classes"]

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
        }

        model = model_functions[model_name](num_classes=num_classes)

        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        return model
    
    def train(
        self,
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        csv_train_log_file=None,
        run_name="",
    ):
        epoch_logs = []

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        """
        ## Define the learning rate schedule
        """

        def lr_schedule(epoch):
            if self.lr_warmup and epoch < 5:
                lr = self.learning_rate * 0.1
            elif epoch < math.ceil(self.epochs * 0.5):
                lr = self.learning_rate
            elif epoch < math.ceil(self.epochs * 0.75):
                lr = self.learning_rate * 0.1
            else:
                lr = self.learning_rate * 0.01

            if epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            return lr

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr_schedule(0),
                weight_decay=0.0001,
                momentum=0.9,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr_schedule(0), eps=1e-07,
            )

        def train_one_epoch():
            running_loss = 0
            running_predicted = []
            running_labels = []

            model.train()

            # Loop through the training dataset and train the model
            for i, (input, label) in enumerate(train_dataloader, 0):
                input, label = input.to(self.device), label.to(self.device)

                output = model(input)
                loss = criterion(output, label)

                # Zero your gradients every batch
                optimizer.zero_grad()

                # Compute the loss and its gradients
                loss.backward()
                # Adjust learning weights
                optimizer.step()

                # Store the running loss for the training dataset
                running_loss += loss

                # Select the largest prediction value for each class
                _, predicted = torch.max(output.data, 1)

                # Store the batch predicted values for the training dataset
                running_predicted.extend(predicted.tolist())
                running_labels.extend(label.tolist())

            # Calucate the training loss by dividing the total loss by number of batches
            epoch_loss = running_loss.item() / len(train_dataloader)

            # Use sklearn to calculate the training accuracy
            epoch_accuracy = accuracy_score(running_labels, running_predicted)

            return epoch_loss, epoch_accuracy

        for epoch in range(epochs):
            print("EPOCH %s/%s:" % (epoch + 1, epochs))

            start_time = datetime.now()

            # Update the learning rate
            lr_schedule(epoch)

            # Train the model for one epoch
            train_loss, train_accuracy = train_one_epoch()

            # Evaluate the model on the validation dataset after each epoch
            validation_loss, validation_accuracy = self.evaluate(
                model, val_dataloader, save_predictions=False, predictions_csv_file=None
            )

            # Calculate the epoch time
            end_time = datetime.now()
            epoch_time = end_time - start_time

            print(
                "EPOCH %s/%s END: TRAIN loss: %s acc: %s VAL loss: %s acc: %s (time: %s)"
                % (
                    epoch + 1,
                    epochs,
                    train_loss,
                    train_accuracy,
                    validation_loss,
                    validation_accuracy,
                    epoch_time,
                )
            )

            epoch_logs.append(
                [
                    epoch,
                    train_accuracy,
                    train_loss,
                    validation_accuracy,
                    validation_loss,
                ]
            )

        if self.save_epoch_logs:
            # Save the epoch log to a csv file
            with open(csv_train_log_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["epoch", "accuracy", "loss", "val_accuracy", "val_loss"]

                writer.writerow(csv_columns)
                writer.writerows(epoch_logs)

        return model

    def evaluate(
        self, model, dataloader, save_predictions=False, predictions_csv_file=None
    ):
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0

        running_predicted = []
        running_labels = []
        predictions = []

        model.eval()

        with torch.inference_mode():
            for data in dataloader:
                # Get the images and labels from the dataloader
                input = data[0].to(self.device, non_blocking=True)
                labels = data[1].to(self.device, non_blocking=True)

                # Calculate the prediction values for each image
                outputs = model(input)

                # Calculate the loss for the batch
                loss += criterion(outputs, labels)

                # Select the largest prediction value for each class
                _, predicted = torch.max(outputs.data, 1)

                # Store the batch predicted values for the dataset
                running_predicted.extend(predicted.tolist())
                running_labels.extend(labels.tolist())

                if save_predictions:
                    # loop through the batch and add each prediction to the predictions list
                    for output in outputs:
                        predictions.append(output.tolist())

        if save_predictions:
            # Add the true values to the first column and the predicted values to the second column
            true_and_pred = np.vstack((running_labels, running_predicted)).T

            # Add each label predictions to the true and predicted values
            csv_output_array = np.concatenate((true_and_pred, predictions), axis=1)

            # Save the predictions to a csv file
            with open(predictions_csv_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["true_value", "predicted_value"]
                for i in range(len(predictions[0])):
                    csv_columns.append("label_" + str(i))

                writer.writerow(csv_columns)
                writer.writerows(csv_output_array.tolist())

        # Calucate the validation loss by dividing the total loss by number of batches
        validation_loss = loss.item() / len(dataloader)

        # Use sklearn to calculate the validation accuracy
        validation_accuracy = accuracy_score(running_labels, running_predicted)

        return [validation_loss, validation_accuracy]

    def save(self, model, model_path):
        torch.save(model, model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        model.eval()

        return model
