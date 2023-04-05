import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# image_folder_path = "test/images/"
# label_folder_path = "test/labels/"

# image_names = path_list = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder_path) if filename.endswith(".jpg")]
#
# image = image_names[10]
#
# # Load the image
# img = plt.imread(image_folder_path+image+".jpg")
# height, width, _ = img.shape
#
# # Load the bounding box information
# with open(label_folder_path+image+".txt") as f:
#     bboxes = f.readlines()
#
# # Plot the image
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(img)
#
# # Plot the bounding boxes
# for bbox in bboxes:
#     bbox = bbox.strip().split(" ")
#     class_name = bbox[0]
#     x, y, w, h = [float(x) for x in bbox[1:5]]
#     x = int(x * width)
#     y = int(y * height)
#     w = int(w * width)
#     h = int(h * height)
#     rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
# plt.show()


# # Plot the bounding boxes
# min_confidence = 1.0
# uncertain_bbox = None
# for bbox in bboxes:
#     bbox = bbox.strip().split(" ")
#     class_name = bbox[0]
#     x, y, w, h, confidence = [float(x) for x in bbox[1:]]
#     x = int(x * width)
#     y = int(y * height)
#     w = int(w * width)
#     h = int(h * height)
#     if confidence < min_confidence:
#         min_confidence = confidence
#         uncertain_bbox = (x, y, w, h)
#     rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#
# # Plot the most uncertain bounding box
# if uncertain_bbox is not None:
#     x, y, w, h = uncertain_bbox
#     rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
#     ax.add_patch(rect)
#     ax.text(x, y, "uncertain", color='b')
#
# plt.show()


def plot_bboxes(image_folder_path, label_folder_path, image, extension=".jpg"):
    # Load the image
    img = plt.imread(image_folder_path+image+extension)
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
