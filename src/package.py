import glob
import os

import click
import pandas


@click.command()
@click.option('--result_folder', type=str, default='../results')
@click.option('--version', type=str, default="flash-holmes")
def main(
        result_folder, version
):
    data_folder = os.path.abspath("../data")
    result_folder = os.path.abspath(result_folder)

    holmes_results = pandas.read_csv(f"{data_folder}/leaderboards/{version}.csv", index_col=0)
    holmes_results = holmes_results[holmes_results["train portion"] == 0.03125]

    result_frame = pandas.read_csv(f"{result_folder}/results_{version}.csv", index_col=0)

    result_frame["model_name"] = result_frame.apply(lambda row: f"{row['model_name']}-{row['encoding']}", axis=1)

    for num_hidden_layers, grouped_results in result_frame.groupby("num_hidden_layers"):
        print(f"Package results for {num_hidden_layers} hidden layers")

        package_results = holmes_results.copy()
        for model_name, model_frame in grouped_results.groupby("model_name"):
            average_scores = {
                dataset: dataset_frame["score"].mean()
                for dataset, dataset_frame in model_frame.groupby("probing_dataset")
            }
            result_column = pandas.Series(
                data = [
                    average_scores.get(dataset, -1)
                    for dataset in holmes_results["probing dataset"]
                ],
                index = holmes_results.index
            )
            package_results.insert(
                loc = len(holmes_results.columns) - 1,
                column = model_name,
                value = result_column
            )
        package_results.to_csv(f"{result_folder}/packaged_results_{version}_{num_hidden_layers}_hidden_layers.csv")



if __name__ == "__main__":
    main()