import argparse, csv, os, sys, yaml
from datetime import datetime

script_version = "1.0.2"


def get_dataset_details(dataset_name):
    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cats_vs_dogs": {
            "train_path": "./cats_vs_dogs/train/",
            "val_path": "./cats_vs_dogs/val/",
            "test_path": "./cats_vs_dogs/test/",
            "num_classes": 2,
            "dataset_shape": (128, 128, 3),
            "normalization": {
                "mean": (0.4872, 0.4544, 0.4165),
                "std": (0.2622, 0.256, 0.2584),
            },
        },
        "cifar10": {
            "train_path": "./cifar10/train/",
            "val_path": "./cifar10/val/",
            "test_path": "./cifar10/test/",
            "num_classes": 10,
            "dataset_shape": (32, 32, 3),
            "normalization": {
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.247, 0.2435, 0.2616),
            },
        },
        "cifar100": {
            "train_path": "./cifar100/train/",
            "val_path": "./cifar100/val/",
            "test_path": "./cifar100/test/",
            "num_classes": 100,
            "dataset_shape": (32, 32, 3),
            "normalization": {
                "mean": (0.5071, 0.4865, 0.4409),
                "std": (0.2673, 0.2564, 0.2762),
            },
        },
        "imagenette": {
            "train_path": "./imagenette/train/",
            "val_path": "./imagenette/val/",
            "test_path": "./imagenette/test/",
            "num_classes": 10,
            "dataset_shape": (224, 224, 3),
            "normalization": {
                "mean": (0.4623, 0.458, 0.4305),
                "std": (0.2829, 0.2797, 0.3018),
            },
        },
        "uc_merced": {
            "train_path": "./uc_merced/train/",
            "val_path": "./uc_merced/val/",
            "test_path": "./uc_merced/test/",
            "num_classes": 21,
            "dataset_shape": (224, 224, 3),
            "normalization": {
                "mean": (0.4829, 0.4892, 0.4489),
                "std": (0.2184, 0.2024, 0.1959),
            },
        },
    }

    return datasets[dataset_name]


def image_classification(
    machine_learning_framework="TensorFlow",
    model_name="ResNet20",
    dataset_name="cifar10",
    op_determinism=False,
    batch_size=128,
    learning_rate=0.001,
    lr_scheduler=False,
    lr_warmup=False,
    optimizer="SGD",
    epochs=0,
    nestrov=False,
    seed_val=1,
    run_name="",
    start="",
    save_model=False,
    save_predictions=False,
    save_epoch_logs=False,
    save_tensorboard_logs=False,
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()

    elif machine_learning_framework == "PyTorch":
        from pytorch_framework import Pytorch

        framework = Pytorch()

    framework.learning_rate = learning_rate
    framework.lr_scheduler = lr_scheduler
    framework.lr_warmup = lr_warmup
    framework.optimizer = optimizer
    framework.epochs = epochs
    framework.nesterov = nestrov
    framework.save_epoch_logs = save_epoch_logs
    framework.save_tensorboard_logs = save_tensorboard_logs

    if seed_val != 1:
        """
        ## Configure framework for fixed seed runs
        """
        framework.set_random_seed(seed_val)

    if op_determinism:
        """
        ## Configure framework deterministic operations
        """
        framework.set_op_determinism()

    """
    ## Get the dataset details
    """
    dataset_details = get_dataset_details(dataset_name)

    """
    ## Load the dataset
    """
    # Always use the same random seed for the dataset
    train_dataset, val_dataset, test_dataset = framework.load_dataset(
        dataset_details, batch_size
    )

    """
    ## Create the model
    """
    model = framework.load_model(model_name, dataset_details)

    """
    ## Create the base name for the log and model files
    """
    base_name = os.path.basename(sys.argv[0]).split(".")[0]

    if len(run_name) >= 1:
        base_name = run_name

    """
    ## Train the model
    """
    # Time the training
    start_time = datetime.now()

    if seed_val != 1:
        csv_train_log_file = "%s_log_%s_%s.csv" % (
            base_name,
            machine_learning_framework,
            seed_val,
        )
    else:
        csv_train_log_file = "%s_log_%s_%s.csv" % (
            base_name,
            machine_learning_framework,
            start,
        )

    # Train the model
    trained_model = framework.train(
        model,
        train_dataset,
        val_dataset,
        epochs,
        csv_train_log_file,
        run_name,
    )

    # Calculate the training time
    end_time = datetime.now()
    training_time = end_time - start_time

    """
    ## Evaluate the trained model and save the predictions
    """
    prediction_path = ""

    if save_predictions:
        if os.path.exists("predictions/") == False:
            os.mkdir("predictions/")

        prediction_path = "predictions/"

        if run_name != "":
            if os.path.exists("predictions/" + run_name + "/") == False:
                os.mkdir("predictions/" + run_name + "/")
            prediction_path = "predictions/" + run_name + "/"

    if seed_val != 1:
        predictions_csv_file = (
            prediction_path + base_name + "_seed_" + str(seed_val) + ".csv"
        )
    else:
        predictions_csv_file = prediction_path + base_name + "_ts_" + start + ".csv"

    score = framework.evaluate(
        trained_model, test_dataset, save_predictions, predictions_csv_file
    )

    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    print("Training time: ", training_time)

    """
    ## Save the model
    """
    if save_model:
        if os.path.exists("models/") == False:
            os.mkdir("models/")

        model_path = "models/"

        if run_name != "":
            if os.path.exists("models/" + run_name + "/") == False:
                os.mkdir("models/" + run_name + "/")
            model_path = "models/" + run_name + "/"

        if seed_val != 1:
            model_path = model_path + base_name + "_seed_" + str(seed_val)
        else:
            model_path = model_path + base_name + "_ts_" + start

        # Append the file extension based on the machine learning framework
        if machine_learning_framework == "TensorFlow":
            model_path = model_path + ".keras"
        elif machine_learning_framework == "PyTorch":
            model_path = model_path + ".pth"

        framework.save(trained_model, model_path)

    return score[0], score[1], training_time


def get_system_info(filename):
    if os.path.exists("system_info.py"):
        import system_info

        sysinfo = system_info.get_system_info()

        with open("%s.yaml" % filename, "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


def save_score(
    test_loss,
    test_accuracy,
    machine_learning_framework,
    batch_size,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    optimzer,
    epochs,
    nestrov,
    training_time,
    model_name,
    dataset_name,
    op_determinism,
    seed_val,
    filename,
    run_name="",
    start="",
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()
        
    elif machine_learning_framework == "PyTorch":
        from pytorch_framework import Pytorch

        framework = Pytorch()

    csv_file = filename + ".csv"
    write_header = False

    # If determistic is false and the seed value is 1 then the
    # seed value is totally random and we don't know what it is.
    if seed_val == 1:
        seed_val = "random"

    if not os.path.isfile(csv_file):
        write_header = True

    with open(csv_file, "a") as csvfile:
        fieldnames = [
            "run_name",
            "script_version",
            "date_time",
            "fit_time",
            "python_version",
            "machine_learning_framework",
            "framework_version",
            "batch_size",
            "learning_rate",
            "lr_scheduler",
            "lr_warmup",
            "optimizer",
            "epochs",
            "nestrov",
            "model_name",
            "dataset_name",
            "random_seed",
            "op_determinism",
            "test_loss",
            "test_accuracy",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "run_name": run_name,
                "script_version": script_version + "-" + framework.script_version,
                "date_time": start,
                "fit_time": int(training_time.total_seconds()),
                "python_version": sys.version.replace("\n", ""),
                "machine_learning_framework": machine_learning_framework,
                "framework_version": framework.version,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lr_scheduler": lr_scheduler,
                "lr_warmup": lr_warmup,
                "optimizer": optimzer,
                "epochs": epochs,
                "nestrov": nestrov,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "random_seed": seed_val,
                "op_determinism": op_determinism,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--op-determinism",
        dest="op_determinism",
        help="Run with deterministic operations",
        action="store_true",
    )

    parser.add_argument(
        "--seed-val", dest="seed_val", help="Set the seed value", type=int, default=1
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        help="Size of the mini-batches",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        help="Base learning rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        help="Use the learning rate scheduler",
        action="store_true",
    )

    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Use the learning rate warmup of 5 epochs",
        action="store_true",
    )

    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        help="Name of optimizer to use",
        default="SGD",
        choices=[
            "SGD",
            "Adam",
        ],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        help="Number of epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--nestrov",
        dest="nestrov",
        help="Use Nesterov momentum in the optimizer",
        action="store_true",
    )

    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Name of training run",
        default="",
    )

    parser.add_argument(
        "--save-filename",
        dest="save_filename",
        help="filename used to save the results",
        type=str,
        default=str(os.path.basename(sys.argv[0]).split(".")[0]),
    )

    parser.add_argument(
        "--save-model", dest="save_model", help="Save the model", action="store_true"
    )

    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        help="Save the predictions",
        action="store_true",
    )

    parser.add_argument(
        "--save-epoch-logs",
        dest="save_epoch_logs",
        help="Save the accuracy and loss logs for each epoch",
        action="store_true",
    )

    parser.add_argument(
        "--tensorboard",
        dest="save_tensorboard_logs",
        help="Save TensorBoard logs",
        action="store_true",
    )

    parser.add_argument(
        "--ml-framework",
        dest="machine_learning_framework",
        help="Name of Machine Learning framework",
        default="TensorFlow",
        choices=[
            "TensorFlow",
            "PyTorch",
        ],
        required=True,
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="Name of model to train",
        default="ResNet20",
        choices=[
            "DenseNet_k12d40",
            "DenseNet_k12d100",
            "DenseNet_k24d100",
            "DenseNet_bc_k12d100",
            "DenseNet_bc_k24d250",
            "DenseNet_bc_k40d190",
            "DenseNet121",
            "DenseNet169",
            "DenseNet201",
            "DenseNet264",
            "ResNet20",
            "ResNet32",
            "ResNet44",
            "ResNet56",
            "ResNet110",
            "ResNet1202",
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet101",
            "ResNet152",
        ],
        required=True,
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="The dataset to train the model on",
        default="cifar10",
        choices=[
            "cats_vs_dogs",
            "cifar10",
            "cifar100",
            "imagenette",
            "uc_merced",
        ],
        required=True,
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    save_filename = args.save_filename

    system_info = get_system_info(save_filename)
    seed_val = args.seed_val
    epochs = args.epochs

    start = datetime.now().strftime("%Y%m%d%H%M%S%f")

    print(
        "\nImage Classification (%s - %s - %s): [%s]\n======================\n"
        % (
            args.machine_learning_framework,
            args.model_name,
            args.dataset_name,
            seed_val,
        )
    )
    test_loss, test_accuracy, training_time = image_classification(
        machine_learning_framework=args.machine_learning_framework,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        op_determinism=args.op_determinism,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup=args.lr_warmup,
        optimizer=args.optimizer,
        epochs=epochs,
        nestrov=args.nestrov,
        seed_val=seed_val,
        run_name=args.run_name,
        start=start,
        save_model=args.save_model,
        save_predictions=args.save_predictions,
        save_epoch_logs=args.save_epoch_logs,
        save_tensorboard_logs=args.save_tensorboard_logs,
    )
    save_score(
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        machine_learning_framework=args.machine_learning_framework,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup=args.lr_warmup,
        optimzer=args.optimizer,
        epochs=epochs,
        nestrov=args.nestrov,
        training_time=training_time,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        op_determinism=args.op_determinism,
        seed_val=seed_val,
        filename=save_filename,
        run_name=args.run_name,
        start=start,
    )
