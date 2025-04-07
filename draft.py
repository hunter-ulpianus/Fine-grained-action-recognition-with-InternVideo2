import os
import json
import pandas as pd

frame_path = "/media/sdc/datasets/ssv1/train_frames/"
options_path = "/media/sdc/datasets/ssv1/labels/something-something-v1-labels.csv"
label_list_path = "/media/sdc/datasets/ssv1/labels/something-something-v1-train.csv"
output_label_path = "/media/sdc/datasets/ssv1/labels/train_label_options.json"

video_label_list = pd.read_csv(label_list_path, sep=';', header=None, names=['idx', 'description'], index_col=0)

with open(options_path, 'r', encoding='utf-8') as f:
    options = [line.strip() for line in f.readlines()]

total_list = []

for video_idx in os.listdir(frame_path):
    video_dict = {}

    image_list = []
    sub_frame_path = os.path.join(frame_path, video_idx)
    video_label = video_label_list.loc[int(video_idx), 'description']

    for frame_idx in os.listdir(sub_frame_path):
        image_list.append(frame_idx)
    if video_label in options:
        answer = options.index(video_label)
    else:
        print(f"No such label in options: {video_label}")
        answer = -1
    video_dict.update({
        "id": video_idx,
        "image": image_list,
        "caption": video_label,
        "answer": answer,
        "options": options
    })

    total_list.append(video_dict)
    print(f"Append video {video_idx} into list")

with open(output_label_path, 'w') as f:
    json.dump(total_list, f)
    



    
