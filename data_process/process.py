#-*- coding : utf-8-*-

import pandas as pd
import os
import csv
import numpy as np
import subprocess
import docx  # 读取doc/docx文件
import re
import textgrid
import soundfile as sf
import librosa
import cn2an


# bs
def process_bs(source, target):
    blendshape_example_path = "./data/训练集/arkit.csv"
    with open(blendshape_example_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            blendshape_example = row
            break
    # print(blendshape_example, len(blendshape_example))
    assert len(blendshape_example) == 52
    for item in os.listdir(source):
        if item[-4:] == '.csv':
            blendshape_path = os.path.join(source, item)
            data = pd.read_csv(blendshape_path, usecols=blendshape_example)
            data = np.array(data)
            output_blendshape_path = os.path.join(target, item.split('_')[0] + '.csv')
            print(output_blendshape_path)
            with open(output_blendshape_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(blendshape_example)
                writer.writerows(data)


def process_bs_2(source, target):
    blendshape_example = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker',
                          'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
                          'MouthFrownRight', 'MouthDimpleLeft',
                          'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower',
                          'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight',
                          'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
                          'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft',
                          'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft',
                          'NoseSneerRight']
    for item in os.listdir(source):
        if item[-4:] == '.csv':
            blendshape_path = os.path.join(source, item)
            data = pd.read_csv(blendshape_path, usecols=blendshape_example)
            data = np.array(data)
            output_blendshape_path = os.path.join(target, item.split('_')[0] + '.csv')
            print(output_blendshape_path)
            with open(output_blendshape_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(blendshape_example)
                writer.writerows(data)


# wav
def process_wav(source, target):
    for item in os.listdir(source):
        if item[-4:] == '.wav':
            # print(item)
            # subprocess.call(['cp', '-r', os.path.join(source, item), os.path.join(target)])
            subprocess.call(['ffmpeg', '-i', os.path.join(source, item), '-ac', '1', os.path.join(target, item), '-y'])
            src_sig, sr = sf.read(os.path.join(target, item))
            print(str(sr) + ' ' + item)
            if sr != 16000:
                dst_sig = librosa.resample(src_sig, sr, 16000)
                sf.write(os.path.join(target, item), dst_sig, 16000)


# tsv
def process_tsv_1(source, corpus, ref_name, flag):
    total_tsv = {}
    for item in os.listdir(source):
        print(item)
        file = docx.Document(os.path.join(source, item))
        for p in file.paragraphs:
            if p.text.strip() == '': continue
            # print(str(p.text).split('、', 1))
            number, content = str(p.text).split('、', 1)
            if flag == 'trn':
                output_name = str.lower(item[0]) + number
            elif flag == 'val':
                output_name = str.upper(item[0]) + number
            if any(map(str.isdigit, content)):
                content = cn2an.transform(content, "an2cn")
            content = re.sub(r'[^\u4e00-\u9fa5]', '', content)
            content = ' '.join([i for i in content])
            total_tsv[output_name] = content
    for item in os.listdir(ref_name):
        if item[:-4] in total_tsv:  # 判断键是否存在于字典
            print(item)
            subprocess.call(['cp', '-r', os.path.join(ref_name, item), os.path.join(corpus)])
            with open(os.path.join(corpus, item[:-4] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(total_tsv[item[:-4]])
        elif item[:-4][0] + item[:-4][1:].lstrip('0') in total_tsv:  # 删除字符串中的前导零
            print(item)
            subprocess.call(['cp', '-r', os.path.join(ref_name, item), os.path.join(corpus)])
            with open(os.path.join(corpus, item[:-4] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(total_tsv[item[:-4][0] + item[:-4][1:].lstrip('0')])


def process_tsv_2(corpus, corpus_directory_aligned):
    subprocess.call(
        ['mfa', 'align', corpus, 'mandarin_mfa', 'mandarin_mfa', corpus_directory_aligned, '--clean', 'flag'])


def process_tsv_3(target, corpus, corpus_directory_aligned):
    check_list = [i.split('.')[0] for i in os.listdir(corpus_directory_aligned)]
    for i in os.listdir(corpus):
        if i.split('.')[0] not in check_list:
            print(i)

    for item in os.listdir(corpus_directory_aligned):
        TextGrid_path = os.path.join(corpus_directory_aligned, item)
        tg = textgrid.TextGrid()
        tg.read(TextGrid_path)  # 是文件名
        with open(os.path.join(target, item.replace('.TextGrid', '.tsv')), 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            for key in tg.tiers[0]:
                if key.mark == '': continue
                # print(key.minTime)
                # print(key.maxTime)
                # print(key.mark)
                tsv_w.writerow([key.minTime, key.maxTime, key.mark])


# tsv - test B
def process_tsv_1_(source, corpus, ref_name):
    total_tsv = {}
    with open(source, 'r', encoding='gbk') as f:       # utf-8
        for p in f.readlines():
            if p.strip() == '': continue
            # print(str(p.text).split('、', 1))
            name, content = p.strip().split(':', 1)
            if any(map(str.isdigit, content)):
                content = cn2an.transform(content, "an2cn")
            content = re.sub(r'[^\u4e00-\u9fa5]', '', content)
            content = ' '.join([i for i in content])
            total_tsv[name[:-4]] = content
    for item in os.listdir(ref_name):
        if item[:-4] in total_tsv:  # 判断键是否存在于字典
            print(item)
            subprocess.call(['cp', '-r', os.path.join(ref_name, item), os.path.join(corpus)])
            with open(os.path.join(corpus, item[:-4] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(total_tsv[item[:-4]])
        else:
            print('Error!', item[:-4])


if __name__ == '__main__':
    os.makedirs('./data/trn/bs', exist_ok=True)
    os.makedirs('./data/trn/bs_37', exist_ok=True)
    os.makedirs('./data/trn/tsv', exist_ok=True)
    os.makedirs('./data/trn/wav', exist_ok=True)
    os.makedirs('./data/trn/MFA_corpus', exist_ok=True)
    os.makedirs('./data/trn/aligned_corpus', exist_ok=True)

    # training data
    target_trn = './data/trn/'

    trn_bs = os.path.join(target_trn, 'bs')
    trn_bs_37 = os.path.join(target_trn, 'bs_37')
    trn_tsv = os.path.join(target_trn, 'tsv')
    trn_wav = os.path.join(target_trn, 'wav')
    MFA_corpus = os.path.join(target_trn, 'MFA_corpus')
    aligned_corpus = os.path.join(target_trn, 'aligned_corpus')

    root = './data/训练集/'
    root_wav = root_bs = os.path.join(root, 'audio2face_data_for_train')
    root_tsv = os.path.join(root, 'audio2face_text')
    process_bs(root_bs, trn_bs)
    process_bs_2(root_bs, trn_bs_37)
    process_wav(root_wav, trn_wav)
    process_tsv_1(root_tsv, MFA_corpus, ref_name=trn_wav, flag='trn')
    process_tsv_2(MFA_corpus, aligned_corpus)
    process_tsv_3(trn_tsv, MFA_corpus, aligned_corpus)

    print('number of trn_wav: {}'.format(len(os.listdir(trn_wav))),
          'number of MFA_corpus: {}'.format(len(os.listdir(MFA_corpus))),
          'number of aligned_corpus: {}'.format(len(os.listdir(aligned_corpus))))


    os.makedirs('./data/val/bs', exist_ok=True)
    os.makedirs('./data/val/tsv', exist_ok=True)
    os.makedirs('./data/val/wav', exist_ok=True)
    os.makedirs('./data/val/MFA_corpus', exist_ok=True)
    os.makedirs('./data/val/aligned_corpus', exist_ok=True)

    # validation data
    target_val = './data/val/'

    val_bs = os.path.join(target_val, 'bs')
    val_wav = os.path.join(target_val, 'wav')
    val_tsv = os.path.join(target_val, 'tsv')
    root = './data/验证集/'
    root_wav = root_bs = root
    root_tsv = os.path.join(root, '文本')
    MFA_corpus = os.path.join(target_val, 'MFA_corpus')
    aligned_corpus = os.path.join(target_val, 'aligned_corpus')
    process_bs(root_bs, val_bs)
    process_wav(root_wav, val_wav)
    process_tsv_1(root_tsv, MFA_corpus, ref_name=val_wav, flag='val')
    process_tsv_2(MFA_corpus, aligned_corpus)
    process_tsv_3(val_tsv, MFA_corpus, aligned_corpus)

    # # testing data -A
    # target_tst = './data/tst/'
    # root = './data/测试集-A/'
    # tst_wav = os.path.join(target_tst, 'wav')
    # tst_tsv = os.path.join(target_tst, 'tsv')
    # root_wav = root
    # root_tsv = os.path.join(root, '文本')
    # MFA_corpus = os.path.join(target_tst, 'MFA_corpus')
    # aligned_corpus = os.path.join(target_tst, 'aligned_corpus')
    # # process_wav(root_wav, tst_wav)
    # # process_tsv_1(root_tsv, MFA_corpus, ref_name=tst_wav, flag='val')
    # # process_tsv_2(MFA_corpus, aligned_corpus)
    # # process_tsv_3(tst_tsv, MFA_corpus, aligned_corpus)
    #
    # print('number of tst_wav: {}'.format(len(os.listdir(tst_wav))),
    #       'number of MFA_corpus: {}'.format(len(os.listdir(MFA_corpus))),
    #       'number of aligned_corpus: {}'.format(len(os.listdir(aligned_corpus))))

    '''
    # testing data -B
    target_tst = './data/tst_B/'
    root = './data/测试集-B/'
    tst_wav = os.path.join(target_tst, 'wav')
    tst_tsv = os.path.join(target_tst, 'tsv')
    root_wav = os.path.join(root, 'audio')
    root_tsv = os.path.join(root, 'audio_for_B.txt')
    MFA_corpus = os.path.join(target_tst, 'MFA_corpus')
    aligned_corpus = os.path.join(target_tst, 'aligned_corpus')
    # process_wav(root_wav, tst_wav)
    # process_tsv_1_(root_tsv, MFA_corpus, ref_name=tst_wav)
    # process_tsv_2(MFA_corpus, aligned_corpus)
    process_tsv_3(tst_tsv, MFA_corpus, aligned_corpus)

    # print('number of tst_wav: {}'.format(len(os.listdir(tst_wav))),
    #       'number of MFA_corpus: {}'.format(len(os.listdir(MFA_corpus))),
    #       'number of aligned_corpus: {}'.format(len(os.listdir(aligned_corpus))))
    '''