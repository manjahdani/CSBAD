import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_bboxes(image_folder_path, label_folder_path, image):
    # Load the image
    img = plt.imread(image_folder_path+image+".jpg")
    height, width, _ = img.shape

    # Load the bounding box information
    with open(label_folder_path+image+".txt") as f:
        bboxes = f.readlines()

    # Plot the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # Plot the bounding boxes
    min_confidence = 1.0
    uncertain_bbox = None
    for bbox in bboxes:
        bbox = bbox.strip().split(" ")
        cx, cy, w, h, confidence = [float(x) for x in bbox[1:]]
        x = int((cx - w / 2) * width)
        y = int((cy - h / 2) * height)
        w = int(w * width)
        h = int(h * height)
        if confidence < min_confidence:
            min_confidence = confidence
            uncertain_bbox = (x, y, w, h)
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Plot the most uncertain bounding box
    if uncertain_bbox is not None:
        x, y, w, h = uncertain_bbox
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "uncertain", color='b')

    plt.show()
    
    
def plot_results(path="ShareWithKDteam/WALT/inference_results.csv", metric="mAP50", shaded_region=False):

    df = pd.read_csv(path)
    df = df.groupby(['strategy', 'samples']).agg({metric: ['mean', 'std']}).reset_index()
    df.columns = df.columns.droplevel()
    df.columns = ['strategy', 'samples', metric+'_mean', metric+'_std']

    # Remove the rows where strategy is 'yolov8n' or 'yolov8x'
    df_plot = df[~df['strategy'].isin(['yolov8n', 'yolov8x'])]
    fig, ax = plt.subplots()
    for strat, group in df_plot.groupby('strategy'):
        x = group['samples']
        y = group[metric+'_mean']
        yerr = group[metric+'_std']
        if shaded_region:
            ax.plot(x, y, label=strat, marker="o")
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        else:
            ax.errorbar(x, y, yerr=yerr, label=strat, marker="o")
    group = df[df['strategy'] == 'yolov8n']
    y = group[metric+'_mean'].values[0]
    ax.axhline(y, c="k", ls=":", label="Student")
    group = df[df['strategy'] == 'yolov8x']
    y = group[metric+'_mean'].values[0]
    ax.axhline(y, c="r", ls=":", label="Teacher")
    ax.set_xlabel('Samples')
    ax.set_ylabel(metric)
    ax.set_xlim(15, 110)
    ax.set_xticks([25, 50, 75, 100])
    ax.legend()
    plt.show()

def plot_results_multiple(path="ShareWithKDteam/WALT/inference_results.csv", error_bar=False, size=(16, 16)):
    df = pd.read_csv(path)
    df = df.groupby(['strategy', 'samples']).agg(
        {'precision': ['mean', 'std'], 'recall': ['mean', 'std'], 'mAP50': ['mean', 'std'],
         'mAP50-95': ['mean', 'std']}).reset_index()

    # Flatten column names and select relevant columns
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = df[['strategy_', 'samples_', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std', 'mAP50_mean',
             'mAP50_std', 'mAP50-95_mean', 'mAP50-95_std']]

    # Remove the rows where strategy is 'yolov8n' or 'yolov8x'
    df_plot = df[~df['strategy_'].isin(['yolov8n', 'yolov8x'])]

    # Create the subplots
    fig, axs = plt.subplots(2, 2, figsize=size)
    for i, col in enumerate(['precision', 'recall', 'mAP50', 'mAP50-95']):
        ax = axs[i // 2][i % 2]
        for strat, group in df_plot.groupby('strategy_'):
            x = group['samples_']
            y = group[f'{col}_mean']
            yerr = group[f'{col}_std']
            if error_bar:
                ax.errorbar(x, y, yerr=yerr, label=strat, marker="o")
            else:
                ax.plot(x, y, label=strat, marker="o")
        group = df[df['strategy_'] == 'yolov8n']
        y = group[f'{col}_mean'].values[0]
        axs[i // 2, i % 2].axhline(y, c="k", ls=":", label="Student")
        group = df[df['strategy_'] == 'yolov8x']
        y = group[f'{col}_mean'].values[0]
        axs[i // 2, i % 2].axhline(y, c="r", ls=":", label="Teacher")
        ax.set_title(col)
        ax.set_xlabel('Samples')
        ax.set_ylabel(col)
        ax.set_xlim(15, 110)
        ax.set_xticks([25, 50, 75, 100])
        ax.legend()
    plt.show()
