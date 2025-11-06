import os


def upload(
        file_path,
        target_path,
        file_name,
):
    while True:
        os.system("mkdir -p  " + target_path)
        os.system("cp " + file_path + " " + target_path + "/" + file_name)

        if "pred" not in file_name:
            break

        if "pred" in file_name and len(open(target_path + "/" + file_name).readline()) > 1:
            break

