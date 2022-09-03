import os
import pandas as pd
import numpy as np
import csv
from scipy.signal import savgol_filter

root = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3/multimodal_context_checkpoint_166/"

# predict_dir = root + "val/"
# post_process_dir = root + "val_post/"

# predict_dir = root + "tst/"
# post_process_dir = root + "tst_post/"

# predict_dir = root + "tst_B/"
# post_process_dir = root + "tst_B_post/"

predict_dir = "... your path/AIWIN/result/output/infer_sample/my_tst_A_post3/"
post_process_dir = "... your path/AIWIN/result/output/infer_sample/my_tst_A_post/"

os.makedirs(post_process_dir, exist_ok=True)

blendshape_example = []
blendshape_example_path = "... your path/AIWIN/data/训练集/arkit.csv"
with open(blendshape_example_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        blendshape_example = row
        break

# 52减去眼睛、舌头blendshape
keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft',
        'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper',
        'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
        'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']

for item in os.listdir(predict_dir):
    data = pd.read_csv(predict_dir + item, usecols=keys)
    data = np.array(data)

    # smoothing
    n_poses = data.shape[0]
    out_poses = np.zeros((n_poses, data.shape[1]))

    for i in range(data.shape[1]):
        out_poses[:, i] = savgol_filter(data[:, i], 9, 4)  # NOTE: smoothing on rotation matrices is not optimal

    out_poses = np.pad(out_poses, ((0, 0), (14, 1)), 'constant', constant_values=(0, 0))

    print(data.shape)
    out_bvh_path = os.path.join(post_process_dir, item)
    with open(out_bvh_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(blendshape_example)
        writer.writerows(out_poses)

