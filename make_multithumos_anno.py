import json
import os
import csv
import numpy as np
import pandas as pd


def get_class_index_map(class_info_path='datasets/multithumos/class_list.txt'):
    txt = np.loadtxt(class_info_path, dtype=str)
    idx_act = {}
    for l in txt:
        idx_act[int(l[0])] = l[1]
    return idx_act


def get_video_info(video_info_path):
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'fps': info[1],
            'sample_fps': info[2],
            'count': info[3],
            'sample_count': info[4]
        }
    return video_infos


if __name__ == '__main__':
    test_info = get_video_info('datasets/thumos_anno/test_video_info.csv')
    val_info = get_video_info('datasets/thumos_anno/val_video_info.csv')

    f = open('datasets/multithumos/multithumos_test_ours.csv', 'w')
    f2 = open('datasets/multithumos/multithumos_val_ours.csv', 'w')
    wr = csv.writer(f)
    wr.writerow(['video', 'type', 'type_idx', 'start',
                 'end', 'startFrame', 'endFrame'])
    wr2 = csv.writer(f2)
    wr2.writerow(['video', 'type', 'type_idx', 'start',
                  'end', 'startFrame', 'endFrame'])

    mthmous_path = 'datasets/multithumos/multithumos.json'

    with open(mthmous_path, 'r') as file:
        mthmous = json.load(file)
    with open('datasets/multithumos/class_list.txt', 'r') as file:
        class_list = file.readlines()

    total_video_nums = len(list(mthmous.keys()))
    video_names = list(mthmous.keys())

    act_mapping = get_class_index_map()

    # print(class_list)

    for video_name in video_names:
        num_actions = len(mthmous[video_name]['actions'])
        for i in range(num_actions):
            duration = mthmous[video_name]['duration']
            action_number = mthmous[video_name]['actions'][i][0]
            start = float(mthmous[video_name]['actions'][i][1])
            end = float(mthmous[video_name]['actions'][i][2])

            if mthmous[video_name]['subset'] == 'testing':
                fps = test_info[video_name]['fps']
                if int(start*fps) == int(end*fps):
                    continue
                wr.writerow(
                    [video_name, act_mapping[action_number], action_number, start, end, int(start*fps), int(end*fps)])
            else:
                fps = val_info[video_name]['fps']
                if int(start*fps) == int(end*fps):
                    continue
                wr2.writerow(
                    [video_name, act_mapping[action_number], action_number, start, end, int(start*fps), int(end*fps)])

    f.close()
    f2.close()
