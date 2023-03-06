import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


STRATEGIES = ["n_first", "threshLeastconfidenceStreambased_max", "threshTopconfidenceStreambased_max", "uniformStreambased"]
CAM = "cam2"
WEEK = "week4"

# # precision, recall, mAP50, mAP50-95
df = pd.read_csv("/Users/dcac/Data/knowledge_distillation/inference_results.csv")
df = df.groupby(['strategy', 'cam', 'week', 'samples']).agg({'mAP50-95': ['mean']}).reset_index()
df.columns = df.columns.droplevel()
df.columns = ['strategy', 'cam', 'week', 'samples', 'mAP50-95']
df = df[df["week"] == WEEK]
df = df[df["cam"] == CAM]
x_values = [25, 50, 75, 100, 150, 200, 250]
tick_positions = [0, 1, 2, 3, 5, 7, 9]
tick_labels = ['25', '50', '75', '100', '150', '200', '250']
fig, ax = plt.subplots(dpi=300)
for strat, group in df.groupby('strategy'):
    x = group['samples']
    y = group["mAP50-95"]
    if strat == "threshTopconfidenceStreambased_max":
        ax.plot(tick_positions, y, label="Top Confidence", c="limegreen", marker="*", markersize=8)
    if strat == "threshLeastconfidenceStreambased_max":
        ax.plot(tick_positions, y, label="Least Confidence", c="orange", marker="s", markersize=5)
    # if strat == "threshconfidentTeacherStreambased_max":
    #     ax.plot(tick_positions, y, label="Teacher Top Confidence", c="limegreen", marker="+", markersize=8, alpha=0.5)
    if strat == "n_first":
        ax.plot(tick_positions, y, label="$N$-First", c="slateblue", marker="o", markersize=5)
    if strat == "uniformStreambased":
        ax.plot(tick_positions, y, label="Uniform", c="m", marker="d", markersize=5)
    if strat == "yolov8n":
        ax.plot(tick_positions, y, label="Student", marker="", c="k", ls=":")
    if strat == "yolov8x6":
        ax.plot(tick_positions, y, label="Teacher", marker="", c="r", ls=":")
ax.set_xlabel('Samples')
ax.set_ylabel("mAP50-95")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_xlim(-1, 10)
ax.set_title(CAM + " - " + WEEK)
ax.legend()
plt.show()
