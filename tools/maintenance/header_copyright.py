#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

COPYRIGHT_STRING = "Copyright 2020 ETH Zurich. All Rights Reserved."

import argparse
import fileinput
import os
import pathlib

def get_files(path, ext):
    return path.rglob(f"*.{ext}")

def has_header(path, header_start: str):
    with open(path) as f:
        lines = f.readlines()
        if len(lines) == 0:
            return False
        if lines[0].startswith(header_start):
            return True
        return False

def add_header(path, header_start: str):
    if has_header(path, header_start):
        return

    print(f"Adding header to {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    lines.insert(0, header_start + " " + COPYRIGHT_STRING + "\n")

    with open(path, "w") as f:
        content = "".join(lines)
        f.write(content)


def get_all_files(root: str):
    PATH = pathlib.Path(root)
    src = PATH / "src" / "mirheo"

    all_files = []

    for d in ["core", "plugins", "bindings"]:
        folder = src / d
        all_files.extend(get_files(folder, "h"))
        all_files.extend(get_files(folder, "cpp"))
        all_files.extend(get_files(folder, "cu"))

    return all_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default=os.path.join('..','..'), help="root of the project")
    args = parser.parse_args()

    files = get_all_files(args.root)

    for fname in files:
        add_header(fname, "//")
