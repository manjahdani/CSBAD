import os
import argparse
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

MODELS = ['yolov8n', 'yolov8s','yolov8m', 'yolov8l', 'yolov8x6']
MODELS_COLORS = ['k', 'b','g', 'p', 'r']

STUDENT_TEACHER_CONFIG = ['student', 'teacher', 'teacher', 'teacher', 'teacher']

METRICS = ['precision', 'recall', 'mAP50', 'mAP50-95', 'fitness']

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : SMALL_SIZE}

ticks_x = [0, 250, 500, 750, 1000, 1500] # [0, 25, 50, 75, 100]
ticks_y = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

interval_to_show_x = [15, 1300] # [15, 150]
interval_to_show_y = [0.0, 1]

def get_color(label):
    return MODELS_COLORS[MODELS.index(label)]

def get_coco_label(label):
    return STUDENT_TEACHER_CONFIG[MODELS.index(label)] + ':' + label
    
def create_plots(dfs_strategy, dfs_coco, save_path):
    for (metric, df_strat), (_, df_coco) in zip(dfs_strategy.items(), dfs_coco.items()):
        samples = list(df_strat.columns)
        samples.sort()

        for label, values in zip(df_strat.index, df_strat.values):
                plt.plot(samples, values, label = label)

        for label, values in zip(df_coco.index, df_coco.values):
            color = get_color(label)
            label = get_coco_label(label)
            plt.hlines(y = values, xmin = ticks_x[0], xmax = ticks_x[-1]+100, colors = color, linestyles = '--', lw = 2, label = label)

        plt.xticks(ticks = ticks_x)
        plt.yticks(ticks = ticks_y)

        plt.xlim(interval_to_show_x)
        plt.ylim(interval_to_show_y)

        plt.xlabel('Number of selected frames')
        plt.ylabel(metric)

        plt.legend(list(df_strat.index), loc = 'lower right')

        plt.legend()

        plt.savefig(os.path.join(save_path, metric + '.png'))

        plt.show()


def pivot(df):
    dfs = {}
    for metric in METRICS:
        dfs[metric] = pd.pivot_table(df, values=metric, index='strategy', columns='samples',
                                        aggfunc='mean', margins=False, margins_name='mean')
        print('   ', metric)
        print(dfs[metric], end = '\n'*3)
    return dfs


def main(csv_path, save_path):
    csv_path = os.path.abspath(csv_path)

    df = pd.read_csv(csv_path, sep=',')

    strategies = df.strategy.unique()
    samples = df.samples.unique()
    samples.sort()

    print(f'Found following strategies : {strategies}, and following samples {samples}')

    df_strategy = df[['strategy', 'samples', *METRICS]].loc[~df['strategy'].isin(MODELS)]
    dfs_strategy = pivot(df_strategy)
   
    df_coco = df[['strategy', 'samples', *METRICS]].loc[df['strategy'].isin(MODELS)]
    dfs_coco = pivot(df_coco)

    create_plots(dfs_strategy, dfs_coco, save_path)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-c', '--csv_path', type=str, required=True,
                    help='The path to the CSV with the results')
    ap.add_argument('-s', '--save_path', type=str, required=True,
                    help='The path of the folder to save the plots to')
    args = ap.parse_args()

    main(args.csv_path, args.save_path)