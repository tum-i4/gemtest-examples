import argparse
import os
import shlex
import subprocess  # nosec B404
import typing
from pathlib import Path


def sut_folder_loader(folder_path: str, filename="sut.py") -> typing.List[str]:
    path = Path(folder_path)
    sut_files = []
    for x in os.walk(path):
        if filename in x[2]:
            name = x[0] + f"/{filename}"
            sut_files.append(name)
    return sut_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sut_folder",
                        help="Path of the directory where the SUTs are stored.", type=str)
    parser.add_argument("--sut_filename",
                        help="Filename where the SUT can be found", type=str, default="sut.py")
    parser.add_argument("--sut_class",
                        help="Class name of the SUT", type=str, default="SUT")
    sut_args, rest_args = parser.parse_known_args()
    sut_classifier_list = sut_folder_loader(sut_args.sut_folder, sut_args.sut_filename)
    command = "poetry run pytest"
    command_split = shlex.split(command)
    command_split.extend(rest_args)
    for sut_path in sut_classifier_list:
        sut_cli_args = [f"--sut_filepath={sut_path}",
                        f"--sut_class={sut_args.sut_class}"]
        command_split.extend(sut_cli_args)
        cwd = Path(__file__).parent.parent.parent.parent
        subprocess.run(command_split, cwd=cwd, check=False)  # nosec B603
