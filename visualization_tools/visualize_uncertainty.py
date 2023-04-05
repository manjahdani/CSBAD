import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from distillation.pool_based.uncertainty_sampling import strategy_least_confidence
from distillation.visualization.visualize_bounding_boxes import plot_bboxes

subset = strategy_least_confidence("week1/labels_yolov8n_w_conf/", 5, "min")

# for image_name in subset:
#     img = plt.imread("week1/images/"+image_name+".jpg")
#     plt.imshow(img)
#     plt.show()

for image_name in subset:
    plot_bboxes(image_folder_path="week1/images/",
                label_folder_path="week1/labels_yolov8n_w_conf/",
                image=image_name)

# MIN and MAX...