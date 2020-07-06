#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import argparse
import pandas as pd

def dump_csv(fname: str,
             channels: list,
             header: bool):

    df = pd.read_csv(fname)[channels]

    if header:
        print('#', *channels)

    for i, row in df.iterrows():
        print(*[row[c] for c in channels])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help='The csv file.')
    parser.add_argument("channels", type=str, nargs="*", help='List of channels to dump.')
    parser.add_argument("--header", default=False, action='store_true', help='Print header.')
    args = parser.parse_args()

    dump_csv(args.csv_file, args.channels, args.header)
