import os
import pandas as pd
import numpy as np
import csv
from scipy.signal import savgol_filter

# root = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3/multimodal_context_checkpoint_233/"

# predict_dir = root + "val/"
# post_process_dir = root + "val_post3/"

# predict_dir = root + "tst/"
# post_process_dir = root + "tst_post/"

# predict_dir = root + "tst_B/"
# post_process_dir = root + "tst_B_post/"

predict_dir = "... your path/AIWIN/result/output/infer_sample/my_tst_A/"
post_process_dir = "... your path/AIWIN/result/output/infer_sample/my_tst_A_post3/"

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

def sigmoid(z, max_b, min_b, mov, zoom):
    # -5  0.0066928509242848554  5  0.9933071490757153
    # -2.5 0.07585818002124355 2.5 0.9241418199787566
    # -1 0.2689414213699951 1 0.731058578630004
    # -0.5 0.3775406687981454 0.5 0.6224593312018546
    # return min_b + (max_b-min_b) * ((1 / (1 + np.exp(-(z-mov)*10/zoom)))-0.0066928509242848554) * 1/(0.9933071490757153-0.0066928509242848554)
    # return min_b + (max_b - min_b) * ((1 / (1 + np.exp(-(z - mov) * 5 / zoom))) - 0.07585818002124355) * 1 / (0.9241418199787566 - 0.07585818002124355)
    # return min_b + (max_b - min_b) * ((1 / (1 + np.exp(-(z - mov) * 2 / zoom))) - 0.2689414213699951) * 1 / (0.731058578630004 - 0.2689414213699951)
    return min_b + (max_b - min_b) * ((1 / (1 + np.exp(-(z - mov) * 1 / zoom))) - 0.3775406687981454) * 1 / (0.6224593312018546 - 0.3775406687981454)


for item in os.listdir(predict_dir):
    data = pd.read_csv(predict_dir + item, usecols=keys)
    data = np.array(data).transpose()

    out_poses = np.zeros_like(data)
    max_blendshape = [0.365209, 0.150557, 0.152571, 0.707028, 0.706079, 0.772705, 0.778599, 0.361115, 0.364239,
                      0.778734, 0.778406, 0.287064, 0.30077, 0.30789, 0.308011, 0.628363, 0.645024, 0.761167, 0.576951, 0.408798,
                      0.967291, 0.430649, 0.430649, 1, 1, 0.262688, 0.255432, 0.387673, 0.387525, 0.695367, 0.297526,
                      0.297587, 0.072548, 0.369237, 0.380424, 0.220988, 0.243609]
    max_blendshape = np.array(max_blendshape)

    for i in range(len(data)):
        out_poses[i] = sigmoid(data[i], max_b=data[i].max() + (max_blendshape[i]-data[i].max())*0.001, min_b=data[i].min()*0.995, mov=(data[i].max()-data[i].min())/2 + data[i].min(), zoom=(data[i].max()-data[i].min()))


    out_poses = np.pad(out_poses.transpose(), ((0, 0), (14, 1)), 'constant', constant_values=(0, 0))

    print(data.shape)
    out_bvh_path = os.path.join(post_process_dir, item)
    with open(out_bvh_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(blendshape_example)
        writer.writerows(out_poses)