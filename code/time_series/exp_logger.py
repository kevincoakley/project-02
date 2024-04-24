import csv, datetime, os

def exp_logger(path, csv_file, model, dataset, horizon, seed, mse, mae):
 
    csv_path = "%s/%s" % (path, csv_file)

    if os.path.exists(csv_path):
        write_header = False
    else:
        write_header = True

    with open(csv_path, "a") as csvfile:
        fieldnames = [
            "date_time",
            "model",
            "dataset",
            "horizon",
            "seed",
            "mse",
            "mae",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "date_time": datetime.datetime.now(),
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                "seed": seed,
                "mse": mse,
                "mae": mae,
            }
        )