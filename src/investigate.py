import glob
import os

import click

@click.command()
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--version', type=str, default="holmes")
@click.option('--run_probe', type=bool, default=True)
@click.option('--run_mdl_probe', type=bool, default=False)
@click.option('--num_hidden_layers', type=str, default="0")
@click.option('--cuda_visible_devices', type=str)
@click.option('--seeds', type=str, default="0,1,2,3,4")
@click.option('--model_precision', type=str, default="full")
@click.option('--encoding_batch_size', type=str, default=10)
@click.option('--force_encoding', is_flag=True, default=False)
@click.option('--force_probing', is_flag=True, default=False)
@click.option('--dump_preds', is_flag=True, default=False)
@click.option('--control_task_types', type=str, default="none")
@click.option('--in_filter', type=str)
@click.option('--parallel_probing', is_flag=True, default=False)
@click.option('--dump_folder', type=str, default="../dumps")
@click.option('--result_folder', type=str, default="../results")
def main(
        version, model_name, run_probe, run_mdl_probe, num_hidden_layers, cuda_visible_devices,
        seeds, model_precision, encoding_batch_size, force_encoding, force_probing, dump_preds,
        control_task_types, in_filter, parallel_probing, dump_folder, result_folder
):
    failed_runs = []

    dump_folder = os.path.abspath(dump_folder)
    result_folder = os.path.abspath(result_folder)

    os.system(f"mkdir -p {dump_folder}/{version}/")
    os.system(f"mkdir -p {result_folder}/{version}/")

    for control_task_type in control_task_types.split(","):
        for config_file_path in sorted(glob.glob(f"../data/{version}/*/*{control_task_type}*.yaml"), reverse=True):

            if in_filter != None and in_filter + "/" not in config_file_path:
                continue

            encode_command = f"python3 encode.py --dump_folder {dump_folder}/{version} --config_file_path {config_file_path} --model_name {model_name} --model_precision {model_precision} --encoding_batch_size {encoding_batch_size}"

            if force_encoding:
                encode_command += " --force"

            if cuda_visible_devices != None:
                encode_command = "CUDA_VISIBLE_DEVICES=" + cuda_visible_devices + " " + encode_command

            print(f"Run encoding: {encode_command}")

            result = os.system(encode_command)
            if result != 0:
                failed_runs.append(encode_command)

            if parallel_probing:
                probing_command = "python3 probe_parallel.py"
            else:
                probing_command = "python3 probe.py"

            probing_command += f" --dump_folder {dump_folder}/{version}  --result_folder {result_folder}/{version}"
            probing_command += f" --config_file_path {config_file_path} --model_name {model_name} "
            probing_command += f" --run_probe {run_probe}  --run_mdl_probe {run_mdl_probe}"
            probing_command += f" --num_hidden_layers {num_hidden_layers} --seeds {seeds}"

            if dump_preds:
                probing_command += " --dump_preds"
            if force_probing:
                probing_command += " --force"


            if cuda_visible_devices != None:
                probing_command = "CUDA_VISIBLE_DEVICES=" + cuda_visible_devices + " " + probing_command

            print(f"Run probing: {probing_command}")

            result = os.system(probing_command)
            if result != 0:
                failed_runs.append(probing_command)

            if len(failed_runs) > 0:
                file = open(model_name.replace("/", "_") + "_fails.txt", "w+")
                file.writelines(failed_runs)

    probing_command = f"python3 evaluate.py --result_folder {result_folder} --version {version}"
    print(f"Gathering Results: {probing_command}")
    os.system(probing_command)

if __name__ == "__main__":
    main()


