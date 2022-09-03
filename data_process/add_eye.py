import os
import pdb
import csv
import pandas as pd
import numpy as np

root = '... your path/AIWIN/data/trn+val/'

bs_path = root + 'bs/'
dictionary = {}
for item in os.listdir(bs_path):
    if item in ['trn_b24.csv', 'trn_b23.csv', 'trn_b22.csv', 'trn_b21.csv', 'trn_b20.csv', 'trn_b19.csv']:
        print('continue')
        continue
    data = pd.read_csv(os.path.join(bs_path, item))
    data = np.array(data)[:, :14]
    if data.shape[0] not in dictionary:
        dictionary[data.shape[0]] = data

print(sorted(dictionary.keys()), len(dictionary))

'''
[86, 90, 92, 94, 95, 98, 99, 100, 103, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116,
 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 135, 136, 137, 13
8, 140, 141, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,
160, 161, 162, 163, 164, 165, 166, 167, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180
, 181, 183, 184, 186, 188, 189, 190, 192, 193, 195, 196, 197, 198, 199, 200, 202, 203, 2
04, 205, 206, 207, 208, 210, 212, 214, 216, 219, 221, 223, 226, 227, 230, 231, 235, 239,
 241, 244, 246, 250, 251, 256, 260, 267, 268, 271, 279, 280, 291, 296, 310, 321, 341, 41
0]
'''

# 52减去眼睛、舌头blendshape
keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft',
        'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper',
        'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
        'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']

tst_B_bs = "... your path/AIWIN/result/output/infer_sample/my_tst_A_post/"
output_bs = "... your path/AIWIN/result/output/infer_sample/my_tst_A_post_addeye/"

os.makedirs(output_bs, exist_ok=True)

blendshape_example = []
blendshape_example_path = "... your path/AIWIN/data/训练集/arkit.csv"
with open(blendshape_example_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        blendshape_example = row
        break

for item in os.listdir(tst_B_bs):
    data = pd.read_csv(os.path.join(tst_B_bs, item), usecols=keys)
    data = np.array(data)
    assert data.shape[1] == 37
    if data.shape[0] in dictionary:
        data = np.concatenate((dictionary[data.shape[0]], data, np.zeros((data.shape[0], 1))), axis=1)
    else:
        i = 1
        while True:
            frames = data.shape[0] + i
            if frames in dictionary:
                break
            i += 1
        data = np.concatenate((dictionary[frames][:data.shape[0], :], data, np.zeros((data.shape[0], 1))), axis=1)
    assert data.shape[1] == 52
    out_bvh_path = os.path.join(output_bs, item)
    with open(out_bvh_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(blendshape_example)
        writer.writerows(data)


