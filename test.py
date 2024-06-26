import os
import argparse
import torch
from testing import download, inference, inference_coco,plot

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_DIR = os.path.join(BASE_PATH, "testdir")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # necessary
    ap.add_argument("-x", "--run-prefix", type=str, required=True)
    ap.add_argument("-e", "--entity", type=str, required=True)
    ap.add_argument("-p", "--project", type=str, required=True)
    ap.add_argument("-t", "--template", type=str, required=True)
    ap.add_argument("-d", "--dataset_path", type=str, required=True)
    ap.add_argument("-w", "--wandb-download", type=bool, required=False, default=True)
    ap.add_argument("-q", "--query_filter", type=str, required=False, default=None)
    ap.add_argument('-td',"--target_domains", type=str, required=True,help='Set the target_domains to be used for testing')
    ap.add_argument("--test_yolo", action="store_true", default=False,help="Test YOLO models. Default is False",)
    # not important. added to avoid errors in other used scripts
    ap.add_argument("-r", "--runs", nargs="+", required=False)

    args = ap.parse_args()
    print('main argument template',args.template)
    ##################################################################
    if args.wandb_download:
        print("1. Running download script")
        args.folder = os.path.join(DEFAULT_PROJECT_DIR, args.project, "wandb")
        args.list_all = False
        args.list_finished = True
        args.download = True

        download.main(args)  # <- download script
    else:
        print("1. Skipping download script")

    ##################################################################
    print("2. Running inference")
    
    args.weight_path = args.folder
    args.folder = "test"
    args.csv_path = os.path.join(
        DEFAULT_PROJECT_DIR, args.project, "inference_results.csv"
    )

    # Update the following line to call your modified `inference_multistream.main` function
    runs = inference.main(
        args.weight_path,
        args.csv_path,
        args.dataset_path,
        args.run_prefix,
        args.project,
        args.template,
        args.folder,
        args.target_domains.split(',')
    )
    data_names = []
    for run in runs:
        data_names += [run["data-name"] + "->" + run["data"]]
    data_names = list(set(data_names))
    data_names.sort()
    # ###################################################################
    
    if args.test_yolo:
        print("2.1. Running coco inference")
        args.folder = "test"
        inference_coco.main(  # <- coco inference script 
            args.run_prefix,args.csv_path, data_names, args.template, args.folder)


    # print('3. Plotting graphs')
    # plots_dir = os.path.join(DEFAULT_PROJECT_DIR, args.project, "plots")
    # if not os.path.isdir(plots_dir):
    #     os.makedirs(plots_dir)

    # plot.main(args.csv_path, plots_dir)  # <- plotting
