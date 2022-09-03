import argparse
import glob
import os
import pdb
from pathlib import Path
import lmdb
import pyarrow
import pandas as pd
import numpy as np
from utils.data_utils import SubtitleWrapper
from energy import AudioProcesser
import soundfile as sf

blendshape_example = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker',
                      'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
                      'MouthFrownRight', 'MouthDimpleLeft',
                      'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower',
                      'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight',
                      'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
                      'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft',
                      'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft',
                      'NoseSneerRight']

def process_csv(gesture_filename):
    data = pd.read_csv(gesture_filename, usecols=blendshape_example)
    data = np.array(data)
    assert data.shape[1]==len(blendshape_example)
    return data


def make_lmdb_gesture_dataset(path):
    # trn
    base_path = os.path.join(path, 'trn')
    gesture_path = os.path.join(base_path, 'bs')
    audio_path = os.path.join(base_path, 'wav')
    text_path = os.path.join(base_path, 'tsv')
    out_path = os.path.join(base_path, 'lmdb')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B

    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size)]

    # delete existing files
    for i in range(1):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    csv_files = sorted(glob.glob(gesture_path + "/*.csv"))
    save_idx = 0
    for csv_file in csv_files:
        name = os.path.split(csv_file)[1][:-4]
        print(name)
        name = str.lower(name[0]) + name[1:]

        # load subtitles
        # tsv_path = os.path.join(text_path, name + '.tsv')
        # if os.path.isfile(tsv_path):
        #     subtitle = SubtitleWrapper(tsv_path).get()
        # else:
        #     continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path):
            audio_raw, audio_sr = sf.read(wav_path)
            print(audio_raw.shape, audio_sr)
            # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
            # audio_raw = librosa.util.normalize(audio_raw)
            ap = AudioProcesser(wav_path, hop_size=128)  # 320 = 20ms, 16000/hop_size = 50
            energy = ap.get_energy()
            pitch = ap.get_pitch(log=True, norm=False)
            volume = ap.calVolume()
        else:
            continue

        # load skeletons
        poses = process_csv(csv_file)
        # process
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        dataset_idx = 0

        word_list = None
        # word preprocessing
        # word_list = []
        # for wi in range(len(subtitle)):
        #     word_s = float(subtitle[wi][0])
        #     word_e = float(subtitle[wi][1])
        #     word = subtitle[wi][2].strip()
        #
        #     word_tokens = word.split()
        #
        #     for t_i, token in enumerate(word_tokens):
        #         # token = normalize_string(token)
        #         if len(token) > 0:
        #             new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
        #             new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
        #             word_list.append([token, new_s_time, new_e_time])

        # save subtitles and skeletons
        poses = np.asarray(poses, dtype=np.float16)
        clips[dataset_idx]['clips'].append(
            {'words': word_list,
             'poses': poses,
             'audio_raw': audio_raw,
             'energy': energy,
             'pitch': pitch,
             'volume': volume
             })

        # write to db
        for i in range(1):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(save_idx).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)

        all_poses.append(poses)
        save_idx += 1
        print(str(save_idx) + ' / ' + str(len(csv_files)), end='')

    # close db
    for i in range(1):
        db[i].sync()
        db[i].close()

    # val
    base_path = os.path.join(path, 'val')
    gesture_path = os.path.join(base_path, 'bs')
    audio_path = os.path.join(base_path, 'wav')
    text_path = os.path.join(base_path, 'tsv')
    out_path = os.path.join(base_path, 'lmdb')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B

    db = [lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(1):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    csv_files = sorted(glob.glob(gesture_path + "/*.csv"))
    save_idx = 0
    for csv_file in csv_files:
        name = os.path.split(csv_file)[1][:-4]
        print(name)

        # load subtitles
        # tsv_path = os.path.join(text_path, name + '.tsv')
        # if os.path.isfile(tsv_path):
        #     subtitle = SubtitleWrapper(tsv_path).get()
        # else:
        #     continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path):
            audio_raw, audio_sr = sf.read(wav_path)
            print(audio_raw.shape, audio_sr)
            # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
            # audio_raw = librosa.util.normalize(audio_raw)
            ap = AudioProcesser(wav_path, hop_size=128)  # 320 = 20ms, 16000/hop_size = 50
            energy = ap.get_energy()
            pitch = ap.get_pitch(log=True, norm=False)
            volume = ap.calVolume()
        else:
            continue

        # load skeletons
        poses = process_csv(csv_file)
        # process
        clips = [{'vid': name, 'clips': []}]  # validation

        dataset_idx = 0

        word_list = None
        # word preprocessing
        # word_list = []
        # for wi in range(len(subtitle)):
        #     word_s = float(subtitle[wi][0])
        #     word_e = float(subtitle[wi][1])
        #     word = subtitle[wi][2].strip()
        #
        #     word_tokens = word.split()
        #
        #     for t_i, token in enumerate(word_tokens):
        #         # token = normalize_string(token)       # 20220714
        #         if len(token) > 0:
        #             new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
        #             new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
        #             word_list.append([token, new_s_time, new_e_time])

        # save subtitles and skeletons
        poses = np.asarray(poses, dtype=np.float16)
        clips[dataset_idx]['clips'].append(
            {'words': word_list,
             'poses': poses,
             'audio_raw': audio_raw,
             'energy': energy,
             'pitch': pitch,
             'volume': volume
             })

        # write to db
        for i in range(1):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(save_idx).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)

        all_poses.append(poses)
        save_idx += 1
        print(str(save_idx) + ' / ' + str(len(csv_files)), end='')

    # close db
    for i in range(1):
        db[i].sync()
        db[i].close()

    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    print('data mean/std')
    print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


def make_lmdb_gesture_dataset_2(path):
    # '''
    base_path = os.path.join(path, 'trn+val')
    gesture_path = os.path.join(base_path, 'bs')
    audio_path = os.path.join(base_path, 'wav')
    text_path = os.path.join(base_path, 'tsv')
    out_path = os.path.join(base_path, 'lmdb_fix_4')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B

    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    csv_files = sorted(glob.glob(gesture_path + "/*.csv"))
    flag = 0
    cnt = 0
    for v_i, csv_file in enumerate(csv_files):
        name = os.path.split(csv_file)[1][:-4]
        print(name)

        if name in ['trn_b24', 'trn_b23', 'trn_b22', 'trn_b21', 'trn_b20', 'trn_b19']:
            print('continue')
            cnt += 1
            continue

        # name = str.lower(name[0]) + name[1:]

        # load subtitles
        tsv_path = os.path.join(text_path, name + '.tsv')
        if os.path.isfile(tsv_path):
            subtitle = SubtitleWrapper(tsv_path).get()
        else:
            print(name, 'subtitles Error!')
            continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path):
            audio_raw, audio_sr = sf.read(wav_path)
            # print(audio_raw.shape, audio_sr)
            # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
            # audio_raw = librosa.util.normalize(audio_raw)
            ap = AudioProcesser(wav_path, hop_size=128)  # 320 = 20ms, 16000/hop_size = 50
            energy = ap.get_energy()
            pitch = ap.get_pitch(log=True, norm=False)
            volume = ap.calVolume()
        else:
            print(name, 'audio Error!')
            continue

        # load skeletons
        poses = process_csv(csv_file)
        # print(poses.shape)
        # process
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        # split
        if (v_i-cnt) % 5 == 4:        # 五折
            print(name)
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        # word preprocessing
        word_list = []
        for wi in range(len(subtitle)):
            word_s = float(subtitle[wi][0])
            word_e = float(subtitle[wi][1])
            word = subtitle[wi][2].strip()

            word_tokens = word.split()

            for t_i, token in enumerate(word_tokens):
                # token = normalize_string(token)
                if len(token) > 0:
                    new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    word_list.append([token, new_s_time, new_e_time])

        # save subtitles and skeletons
        poses = np.asarray(poses, dtype=np.float16)
        clips[dataset_idx]['clips'].append(
            {'words': word_list,
             'poses': poses,
             'audio_raw': audio_raw,
             'energy': energy,
             'pitch': pitch,
             'volume': volume
             })

        # write to db
        if 'trn_' not in name:
            for kk in range(3):
                for i in range(2):
                    with db[i].begin(write=True) as txn:
                        if len(clips[i]['clips']) > 0:
                            k = '{:010}'.format(flag).encode('ascii')
                            v = pyarrow.serialize(clips[i]).to_buffer()
                            txn.put(k, v)
                flag += 1
        else:
            for i in range(2):
                with db[i].begin(write=True) as txn:
                    if len(clips[i]['clips']) > 0:
                        k = '{:010}'.format(flag).encode('ascii')
                        v = pyarrow.serialize(clips[i]).to_buffer()
                        txn.put(k, v)
            flag += 1

        all_poses.append(poses)
        print(str(v_i) + ' / ' + str(len(csv_files)), end='')

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    print('data mean/std')
    print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


if __name__ == '__main__':
    '''
    python My/scripts/aiwin_dataset_to_lmdb.py ./data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()

    make_lmdb_gesture_dataset(args.db_path)     # process train then test
    # make_lmdb_gesture_dataset_2(args.db_path)       # mix train and test
