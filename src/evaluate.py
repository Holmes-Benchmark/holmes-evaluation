import glob

import click
import pandas


@click.command()
@click.option('--result_folder', type=str, default='../results')
@click.option('--version', type=str, default="holmes-datasets")
def main(
        result_folder, version
):
    result_files = glob.glob(f"{result_folder}/{version}*/**/done/*.csv", recursive=True)
    results = []

    for result_file in result_files:
        metrics = pandas.read_csv(result_file)

        if "full test pearson" in metrics.columns:
            final_metric = list(metrics["full test pearson"])[-1]
        else:
            final_metric = list(metrics["full test f1"])[-1]

        _, probing_dataset, model_name, encoding, control_task_type, sample_size, seed, num_hidden_layers, _ = (
            result_file
                .replace(f"{result_folder}/{version}", "")
                .split("done")[0].split("/")
        )

        probing_dataset = probing_dataset.replace("flash-holmes-", "").replace("holmes-", "")

        results.append({
            "probing_dataset": probing_dataset,
            "model_name": model_name,
            "encoding": encoding,
            "control_task_type": control_task_type,
            "sample_size": sample_size,
            "seed": seed,
            "num_hidden_layers": num_hidden_layers,
            "score": final_metric
        })

    result_frame = pandas.DataFrame(results)

    result_frame.to_csv(f"{result_folder}/results_{version}.csv")


if __name__ == "__main__":
    main()