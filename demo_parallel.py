import numpy as np
import os
import io
import cv2

import torch

import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import logging

os.environ["PYTHONPATH"] = "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality"

from demo_config import (Config,
                         eval_dict_leaf)

from demo.utils import (retrieve_text,
                        _frame_from_video,
                        setup_internvideo2)

num_gpus = 4
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
test_video_folder = "/media/sdc/datasets/ssv2/test_data"
cfg_path = "/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py"
text_candidates_path = "/media/sdc/datasets/ssv2/action_candidates.json"
actual_label_path = "/media/sdc/datasets/ssv2/annotations/test-answers.csv"
log_path = "/media/sdc/fe/planb_log/pure_test1.log"

with open(text_candidates_path, 'r', encoding='utf-8') as f:
    text_candidates = json.load(f)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ])

def find_test_label(label_path, video_id):
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                id_label = line.strip().split(';')
                if id_label[0] == video_id:
                    return id_label[1]
    except FileNotFoundError:
        logging.error(f"file not found: {label_path}")
    return None

def cal_accuracy(text, gt):
    if gt in text:
        error_data = 0
        acc_5 = 1
        if gt == text[0]:
            acc_1 = 1
        else:
            acc_1 = 0
    elif gt == None:
        acc_5 = 0
        acc_1 = f
        error_data = 1
    else:
        error_data = 0
        acc_5 = 0
        acc_1 = 0
    return acc_5, acc_1, error_data

def process_video(video_path, gpu_id, intern_model, tokenizer, cfg):
    torch.cuda.synchronize()
    
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    device = torch.device('cuda:{}'.format(gpu_id))

    text, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=cfg, device=device)
    gt = find_test_label(actual_label_path, os.path.basename(video_path).split('.')[0])

    accuracy_top5, accuracy_top1, error_data= cal_accuracy(text, gt)

    top5_accuracy_list = []
    top1_accuracy_list = []
    
    top5_accuracy_list.append(accuracy_top5)
    top1_accuracy_list.append(accuracy_top1)
    result_list = list(zip(top5_accuracy_list, top1_accuracy_list))

    if gpu_id == 0:
        print(f'(device[{gpu_id}])- Processing video: {os.path.basename(video_path)}\nTop5 predicted labels: {text}, Actual label: {gt}, Accuracy: {int(gt in text)}\n')
    
    return result_list

def dist_files(video_files, gpu_id):
    gpu_results = []
    torch.cuda.set_device(gpu_id)
    processing_count = 0
    cfg = Config.from_file(cfg_path)
    cfg = eval_dict_leaf(cfg)

    intern_model, tokenizer = setup_internvideo2(cfg)

    for file in video_files:
        video_path = os.path.join(test_video_folder, file)
        start_time = datetime.datetime.now()
        gpu_results.append(process_video(video_path, gpu_id, intern_model, tokenizer, cfg))

        if gpu_id == 0:
            print("start_time:", start_time)
            processing_count += 1
            logging.info("Processing video: {}/{}".format(processing_count*num_gpus, len(video_files)*num_gpus))
            current_time = datetime.datetime.now()
            print("current_time:", current_time)
            time_elapsed = current_time - start_time
            average_time_per_video = time_elapsed
            print("average_time_per_video:", average_time_per_video)
            remaining_videos = len(video_files) - processing_count
            estimated_remaining_time = average_time_per_video * remaining_videos

            logging.info("Estimated remaining time: {}".format(estimated_remaining_time))

    logging.info("____________________________________________________________")
    logging.info("GPU[{}: completed processing!]".format(gpu_id))
    logging.info("____________________________________________________________")
    return gpu_results

def distribute_tasks_to_gpus(video_files):
    num_files = len(video_files)
    files_per_gpu = num_files // num_gpus
    indices = [(i * files_per_gpu, (i + 1) * files_per_gpu if i < num_gpus - 1 else num_files) for i in range(num_gpus)]
    param_list = [(video_files[start_idx:end_idx], i) for i, (start_idx, end_idx) in  enumerate(indices)]

    with mp.get_context('spawn').Pool(processes=num_gpus) as excutor:
        results = excutor.starmap(dist_files, param_list)

    return results

def main():
    start_time = datetime.datetime.now()
    video_files = os.listdir(test_video_folder)
    video_data = []
    video_count = len(video_files)

    results = distribute_tasks_to_gpus(video_files)
    flattened_text_ori = [item for sublist in results for item in sublist]
    flattened_text = [item for sublist in flattened_text_ori for item in sublist]
    total_sum = float(sum([item[0] for item in flattened_text]))
    
    top1_num = float(sum([item[1] for item in flattened_text]))
    top5_accuracy = total_sum / video_count
    top1_accuracy = top1_num / video_count
    logging.info(f"Overall top5 accuracy: {top5_accuracy:.4f}({total_sum}/{video_count}, top1 accuracy: {top1_accuracy:.4f}({top1_num}/{video_count}))")

    end_time = datetime.datetime.now()
    total_time_elapsed = end_time - start_time
    logging.info(f"Total time elapsed: {total_time_elapsed}")
    
    return

if __name__ == "__main__":
    main()