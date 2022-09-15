import sys
import os
import csv
import cv2

sys.path.append(os.path.dirname('yolov5'))
from yolov5.val import *


def main(weights_path, csv_path, append_mode, dataset_path, coco):
    mode = 'a' if append_mode else 'w'
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(['name', 'camera', 'COCO', 'mAP 0.5 (640x640)', 'mAP 0.5:95 (640x640)'])

        weights_list = os.listdir(args.weights_path)
        for i, weight in enumerate(weights_list):
            weight_path = os.path.join(weights_path, weight)

            if coco:
                cameras = [folder for folder in os.listdir(dataset_path) if
                           os.path.isdir(os.path.join(dataset_path, folder))]
                for j, camera in enumerate(cameras):
                    camera_path = dataset_path + '/' + camera
                    print(f'[{i + 1}/{len(weights_list)}] [{j + 1}/{len(cameras)}] Evaluating {weight} on {camera}')
                    map05, map0595 = eval_test(weight_path, camera_path, native_res=False)
                    writer.writerow([weight.split('.')[0], camera, 'True', map05, map0595])
            else:
                camera = weight[9:16]
                camera_path = dataset_path + '/' + camera

                print(f'[{i + 1}/{len(weights_list)}] Evaluating {weight}')
                map05, map0595 = eval_test(weight_path, camera_path, native_res=False)
                writer.writerow([weight.split('.')[1], camera, 'False', map05, map0595])


def eval_test(weight_path, camera_path, native_res=False):
    yml = './testing/test_trail22kd.yaml'

    if native_res:
        # Read first image to check the native resolution
        path_test_images = os.path.join(camera_path, 'test', 'images')
        list_images = os.listdir(path_test_images)
        img = cv2.imread(os.path.join(path_test_images, list_images[0]))
        imgsz = img.shape[1]
    else:
        imgsz = 640

    # Read the lines from YAML and change the camera path to evaluate
    with open(yml, 'r') as y:
        lines = y.readlines()
        lines[3] = f'path: {camera_path}\n'

    # Edit file by writing the modified lines
    with open(yml, 'w') as y:
        y.writelines(lines)

    # Run val.py with our parameters
    results = run(data=yml,
                  weights=weight_path,
                  batch_size=1,
                  conf_thres=0.4,
                  iou_thres=0.7,
                  task='test',
                  single_cls=True,
                  imgsz=imgsz)

    # Extract mAP 0.5 and mAP à.5:0.95
    return results[0][2], results[0][3]


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--weights_path', type=str, required=True,
                    help='The path to the weights to be evaluated')
    ap.add_argument('-c', '--csv_path', type=str, required=True,
                    help='The path to the CSV with the results')
    ap.add_argument('-a', '--append', action='store_true',
                    help='Whether the CSV should be append or overwritten')
    ap.add_argument('-d', '--dataset_path', type=str, required=True,
                    help='The path to the dataset folder, it should contain each camera in a separated folder')
    ap.add_argument('-coco', '--coco', action='store_true',
                    help='Whether the weights use COCO classes')
    args = ap.parse_args()

    if not args.append and os.path.exists(args.csv_path):
        print('Are you sure you want to overwrite the CSV file? [y/N]')
        user_resp = input()
        if user_resp != 'y':
            exit()

    main(args.weights_path, args.csv_path, args.append, args.dataset_path, args.coco)
