import csv
import os
import pdb

import numpy as np
# from fastdtw import fastdtw
import torch
from myfastdtw import myfastdtw
from sklearn.metrics import mean_squared_error

# 61点blendshape
# keys = ['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight',
#         'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
#         'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
#         'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch',
#         'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll']

# 52点blendshape
# keys = ['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight',
#         'EyeWideRight', 'JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
#         'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
#         'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut']

# 52减去眼睛、舌头blendshape
keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft',
        'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper',
        'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
        'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']

'''
python evaluate.py --path "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_4_fix/multimodal_context_checkpoint_189/"
python evaluate.py --path "... your path/AIWIN/result/output/infer_sample/"
'''

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str)
args = parser.parse_args()

# 读取预测文件和参考答案文件
# root = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_1/multimodal_context_checkpoint_226/"
root = args.path
# predict_dir = root + "val/"
# predict_dir = root + "val_post2/"
predict_dir = root + "my_val_post/"

truth_dir = '... your path/AIWIN/data/val/bs/'

n = 0
score = 0
with open(root + 'score.txt', 'w') as f_result:
    for prediction in os.listdir(predict_dir):
        if prediction[0] == '.':continue
        y_pred = []
        y_true = []

        with open(predict_dir + prediction, 'r') as csv_pred:
            reader_pred = csv.DictReader(csv_pred)
            for row in reader_pred:
                r_list = []
                for key in keys:
                    r_list.append(float(row[key]))
                y_pred.append(r_list)

        with open(truth_dir + prediction, 'r') as csv_gt:
            reader_gt = csv.DictReader(csv_gt)
            for row in reader_gt:
                r_list = []
                for key in keys:
                    r_list.append(float(row[key]))
                y_true.append(r_list)
        n += 1
        # distance, _ = fastdtw(np.asarray(y_true), np.asarray(y_pred), dist=mean_squared_error)
        distance= myfastdtw(torch.as_tensor(np.asarray(y_true)).unsqueeze(0), torch.as_tensor(np.asarray(y_pred)).unsqueeze(0)).item()
        score += distance
        print(prediction, ':', distance)
        f_result.write(prediction + ': ' + str(distance) + '\n')
    print('score:', score / n)
    f_result.write('score: ' + str(score / n))
