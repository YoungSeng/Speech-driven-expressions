import os
import pandas as pd
import numpy as np
import csv
# from scipy.signal import savgol_filter

# root1 = "... your path/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_1/multimodal_context_checkpoint_226/"
# predict_dir1 = root1 + "tst_A/"
#
# root2 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_2/multimodal_context_checkpoint_249/"
# predict_dir2 = root2 + "tst_A/"
#
# root3 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3/multimodal_context_checkpoint_233/"
# predict_dir3 = root3 + "tst_A/"
#
# root4 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_3/multimodal_context_checkpoint_394/"
# predict_dir4 = root4 + "tst_A/"
#
# root5 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_4/multimodal_context_checkpoint_224/"
# predict_dir5 = root5 + "tst_A/"
#
# post_process_dir = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_post2/" + "tst_A_post2/"



# root1 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_onehot/multimodal_context_checkpoint_383/"
# predict_dir1 = root1 + "tst_A/"
#
# root2 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_1_onehot/multimodal_context_checkpoint_274/"
# predict_dir2 = root2 + "tst_A/"
#
# root3 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_2_onehot/multimodal_context_checkpoint_318/"
# predict_dir3 = root3 + "tst_A/"
#
# root4 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_3_onehot/multimodal_context_checkpoint_330/"
# predict_dir4 = root4 + "tst_A/"
#
# root5 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_4_onehot/multimodal_context_checkpoint_326/"
# predict_dir5 = root5 + "tst_A/"
#
# post_process_dir = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_4_onehot_post2/" + "tst_A_post2/"


root1 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_post2/"
predict_dir1 = root1 + "tst_A_post2/"

root2 = "... your path/AIWIN/result/output/infer_sample/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_4_onehot_post2/"
predict_dir2 = root2 + "tst_A_post2/"
#
post_process_dir = "... your path/AIWIN/result/output/infer_sample/my_tst_A/"

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

for item in os.listdir(predict_dir1):
    # data1 = pd.read_csv(predict_dir1 + item, usecols=keys)
    # data2 = pd.read_csv(predict_dir2 + item, usecols=keys)
    # data3 = pd.read_csv(predict_dir3 + item, usecols=keys)
    # data4 = pd.read_csv(predict_dir4 + item, usecols=keys)
    # data5 = pd.read_csv(predict_dir5 + item, usecols=keys)
    # data1 = np.array(data1)
    # data2 = np.array(data2)
    # data3 = np.array(data3)
    # data4 = np.array(data4)
    # data5 = np.array(data5)

    data1 = pd.read_csv(predict_dir1 + item, usecols=keys)
    data2 = pd.read_csv(predict_dir2 + item, usecols=keys)
    data1 = np.array(data1)
    data2 = np.array(data2)

    # out_poses = 0.16 * data1 + 0.16 * data2 + 0.16 * data3 + 0.36 * data4 + 0.16 * data5
    # out_poses = 0.21 * data1 + 0.21 * data2 + 0.16 * data3 + 0.21 * data4 + 0.21 * data5

    out_poses = 0.3 * data1 + 0.7 * data2

    out_poses = np.pad(out_poses, ((0, 0), (14, 1)), 'constant', constant_values=(0, 0))

    print(data1.shape)
    out_bvh_path = os.path.join(post_process_dir, item)
    with open(out_bvh_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(blendshape_example)
        writer.writerows(out_poses)
