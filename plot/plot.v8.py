import os
import argparse
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : SMALL_SIZE}

ticks_x = [0, 25, 50, 75, 100]
ticks_y = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
hline_x = [-10,110]
interval_to_show_x = [-2,102]


def main(csv_path):
    df = pd.read_csv(csv_path, sep=',')

    strategies = df.strategy.unique()
    samples = df.samples.unique()
    samples.sort()

    print(f'Found following strategies : {strategies}, and following samples {samples}')



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-c', '--csv_path', type=str, required=True,
                    help='The path to the CSV with the results')
    args = ap.parse_args()

    main(args.csv_path)