import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # precision, recall, mAP50, mAP50-95
# df = pd.read_csv("ShareWithKDteam/WALT/inference_results.csv")
# df = df.groupby(['strategy', 'samples']).agg({'mAP50': ['mean', 'std']}).reset_index()
# df.columns = df.columns.droplevel()
# df.columns = ['strategy', 'samples', 'mAP50_mean', 'mAP50_std']
#
# # Remove the rows where strategy is 'yolov8n' or 'yolov8x'
# df_plot = df[~df['strategy'].isin(['yolov8n', 'yolov8x'])]
# fig, ax = plt.subplots()
# for strat, group in df_plot.groupby('strategy'):
#     x = group['samples']
#     y = group['mAP50_mean']
#     yerr = group['mAP50_std']
#     ax.errorbar(x, y, yerr=yerr, label=strat, marker="o")
#     # ax.plot(x, y, label=strat, marker="o")
#     # ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
# group = df[df['strategy'] == 'yolov8n']
# y = group['mAP50_mean'].values[0]
# ax.axhline(y, c="k", ls=":", label="Student")
# group = df[df['strategy'] == 'yolov8x']
# y = group['mAP50_mean'].values[0]
# ax.axhline(y, c="r", ls=":", label="Teacher")
# ax.set_xlabel('Samples')
# ax.set_ylabel('mAP50')
# ax.set_xlim(15, 110)
# ax.set_xticks([25, 50, 75, 100])
# ax.legend()
# plt.show()


# Read data and group by strategy and samples
# df = pd.read_csv("/Users/dcac/Data/knowledge_distillation/ShareWithKDteam/WALT/inference_results.csv")
for i in range(1, 6):
    df = pd.read_csv("/Users/dcac/Data/knowledge_distillation/ShareWithKDteam/WALT/inference_results.csv")
    name = "cam2-week"+str(i)
    df = df[df["data-name"] == name]
    df = df.groupby(['strategy', 'samples']).agg({'mAP50-95': ['mean', 'std']}).reset_index()
    df.columns = df.columns.droplevel()
    df.columns = ['strategy', 'samples', 'mAP50_mean', 'mAP50_std']

    # Remove the rows where strategy is 'yolov8n' or 'yolov8x'
    df_plot = df[~df['strategy'].isin(['yolov8n', 'yolov8x'])]
    fig, ax = plt.subplots()
    for strat, group in df_plot.groupby('strategy'):
        x = group['samples']
        y = group['mAP50_mean']
        yerr = group['mAP50_std']
        ax.errorbar(x, y, yerr=yerr, label=strat, marker="o")
        # ax.plot(x, y, label=strat, marker="o")
        # ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    group = df[df['strategy'] == 'yolov8n']
    y = group['mAP50_mean'].values[0]
    ax.axhline(y, c="k", ls=":", label="Student")
    group = df[df['strategy'] == 'yolov8x']
    y = group['mAP50_mean'].values[0]
    ax.axhline(y, c="r", ls=":", label="Teacher")
    ax.set_title(name)
    ax.set_xlabel('Samples')
    ax.set_ylabel('mAP50-95')
    ax.set_xlim(15, 110)
    ax.set_xticks([25, 50, 75, 100])
    ax.legend()
    plt.show()

# df = df.groupby(['strategy', 'samples']).agg({'precision': ['mean', 'std'], 'recall': ['mean', 'std'], 'mAP50': ['mean', 'std'], 'mAP50-95': ['mean', 'std']}).reset_index()
#
# # Flatten column names and select relevant columns
# df.columns = ['_'.join(col).strip() for col in df.columns.values]
# df = df[['strategy_', 'samples_', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std', 'mAP50_mean', 'mAP50_std', 'mAP50-95_mean', 'mAP50-95_std']]
#
# # Remove the rows where strategy is 'yolov8n' or 'yolov8x'
# df_plot = df[~df['strategy_'].isin(['yolov8n', 'yolov8x'])]
#
# # Create the subplots
# fig, axs = plt.subplots(2, 2, figsize=(16, 16))
# for i, col in enumerate(['precision', 'recall', 'mAP50', 'mAP50-95']):
#     ax = axs[i // 2][i % 2]
#     for strat, group in df_plot.groupby('strategy_'):
#         x = group['samples_']
#         y = group[f'{col}_mean']
#         yerr = group[f'{col}_std']
#         ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
#         ax.plot(x, y, label=strat)
#     group = df[df['strategy_'] == 'yolov8n']
#     y = group[f'{col}_mean'].values[0]
#     axs[i//2, i%2].axhline(y, c="k", ls=":", label="Student")
#     group = df[df['strategy_'] == 'yolov8x']
#     y = group[f'{col}_mean'].values[0]
#     axs[i//2, i%2].axhline(y, c="r", ls=":", label="Teacher")
#     ax.set_title(col)
#     ax.set_xlabel('Samples')
#     ax.set_ylabel(col)
#     ax.set_xlim(15, 110)
#     ax.set_xticks([25, 50, 75, 100])
#     ax.legend()
#
# plt.show()
