import argparse
import math
import pdb
import pprint
import time
import os
import numpy as np
import torch
import utils
from utils.data_utils import SubtitleWrapper
from utils.train_utils import set_logger
from data_loader.data_preprocessor import DataPreprocessor

import random
from energy import AudioProcesser
# import librosa
import soundfile as sf
from transformers import BertTokenizer, BertModel
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_path = "... your path/AIWIN/chinese-roberta-wwm-ext/"
tokenizer_bert = BertTokenizer.from_pretrained(bert_path)
model_bert = BertModel.from_pretrained(bert_path)
text_bert = ""
encoded_input_bert = tokenizer_bert(text_bert, return_tensors='pt')
SOS_token = encoded_input_bert['input_ids'].squeeze()[0]
EOS_token = encoded_input_bert['input_ids'].squeeze()[1]


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentence_to_emotion(words):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from scipy.special import softmax
    MODEL = "... your path/GENEA/genea_challenge_2022/baselines/Tri/cache_text/"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model_ = AutoModelForSequenceClassification.from_pretrained(MODEL)
    n_sentence = ' '.join([i[0] for i in words])
    text = preprocess(n_sentence)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model_(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores


def one_hot(x):
    x = x.unsqueeze(-1)
    condition = torch.zeros(x.shape[0], 2).scatter_(1, x.type(torch.LongTensor), 1)
    return condition


def main(checkpoint_path, transcript_path, wav_path, vid=None):
    args, generator, loss_fn, speaker_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path, device)
    generator.eval()
    pprint.pprint(vars(args))

    # vid = random.sample(range(0, speaker_model.n_words), 1)[0]  # for trimodal
    save_path_modelname = checkpoint_path.split('/')[6]
    save_path_modelepoch = checkpoint_path[:-4].split('/')[-1]
    save_path = '... your path/AIWIN/result/output/infer_sample/' + save_path_modelname + '/' + save_path_modelepoch + '/tst_A/'
    os.makedirs(save_path, exist_ok=True)

    root = "... your path/AIWIN/data/tst/"
    tsv_path = root + 'tsv/'
    wave_path = root + 'wav/'
    tmp_list = os.listdir(tsv_path)

    for item in tmp_list:
        transcript_path = tsv_path + item
        wav_path = wave_path + item[:-4] + '.wav'

        # prepare input
        transcript = SubtitleWrapper(transcript_path).get()
        word_list = []
        for wi in range(len(transcript)):
            word_s = float(transcript[wi][0])
            word_e = float(transcript[wi][1])
            word = transcript[wi][2].strip()

            word_tokens = word.split()

            for t_i, token in enumerate(word_tokens):
                if len(token) > 0:
                    new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    word_list.append([token, new_s_time, new_e_time])

        # inference
        # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
        audio_raw, audio_sr = sf.read(wav_path)
        print(audio_raw.shape, audio_sr)
        ap = AudioProcesser(wav_path, hop_size=128)  # 320 = 20ms, 16000/hop_size = 50
        energy = ap.get_energy()
        pitch = ap.get_pitch(log=True, norm=False)
        volume = ap.calVolume()

        out_list = []
        n_frames = args.n_poses
        clip_length = len(audio_raw) / audio_sr
        print(clip_length)
        pre_seq = torch.zeros((1, n_frames, len(args.data_mean) + 1))       # 20220627

        unit_time = args.n_poses / args.motion_resampling_framerate
        stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
        if clip_length < unit_time:
            num_subdivision = 1
        else:
            num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
        audio_sample_length = int(unit_time * audio_sr)
        pitch_sample_length = int(unit_time * 16000 * 7876 / 2778300)
        # prepare speaker input
        if args.z_type == 'speaker':
            if not vid:
                vid = random.randrange(generator.z_obj.n_words)
            print('vid:', vid)
            vid = torch.LongTensor([vid]).to(device)
        else:
            vid = None
        # print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

        out_dir_vec = None
        start = time.time()
        for i in range(0, num_subdivision):
            start_time = i * stride_time
            end_time = start_time + unit_time
            audio_start = math.floor(start_time / clip_length * len(audio_raw))
            audio_end = audio_start + audio_sample_length
            in_audio = audio_raw[audio_start:audio_end]

            pitch_start = math.floor(start_time / clip_length * len(pitch))
            pitch_end = pitch_start + pitch_sample_length
            in_pitch = pitch[pitch_start:pitch_end]

            in_energy = energy[pitch_start:pitch_end]

            in_volume = volume[pitch_start:pitch_end]

            if len(in_audio) < audio_sample_length:
                if i == num_subdivision - 1:
                    end_padding_duration = audio_sample_length - len(in_audio)
                in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
                in_pitch = np.pad(in_pitch, (0, pitch_sample_length - len(in_pitch)), 'constant')
                in_energy = np.pad(in_energy, (0, pitch_sample_length - len(in_energy)), 'constant')
                in_volume = np.pad(np.squeeze(in_volume), (0, pitch_sample_length - len(in_volume)), 'constant')
            in_audio = torch.as_tensor(in_audio).unsqueeze(0).to(device).float()

            in_energy = torch.as_tensor(in_energy).unsqueeze(0).to(device).float()
            in_pitch = torch.as_tensor(in_pitch).unsqueeze(0).to(device).float()
            in_volume = torch.as_tensor(in_volume).squeeze().unsqueeze(0).to(device).float()

            # prepare text input
            word_seq = DataPreprocessor.get_words_in_time_range(word_list=word_list, start_time=start_time, end_time=end_time)
            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            word_indices = np.zeros(len(word_seq) + 2)
            word_indices[0] = SOS_token
            word_indices[-1] = EOS_token
            frame_duration = (end_time - start_time) / n_frames
            for w_i, word in enumerate(word_seq):
                # print(word[0], end=', ')
                idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
                extended_word_indices[idx] = tokenizer_bert(word[0], return_tensors='pt')['input_ids'].squeeze()[1]
            in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)

            one_hot_embedding_1 = torch.FloatTensor([[0, 1]]).unsqueeze(0).to(device)
            one_hot_embedding = torch.repeat_interleave(one_hot_embedding_1, repeats=55, dim=1)

            if not args.use_emo:
                # print(in_audio.shape)
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid, in_pitch, in_energy, in_volume, None, None, one_hot_embedding=one_hot_embedding)

            elif args.use_txt_emo and not args.use_audio_emo:
                text_emo = torch.as_tensor(np.array(sentence_to_emotion(word_seq))).unsqueeze(0).to(device).float()
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid, in_pitch, in_energy, in_volume, None, text_emo)

            out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

            out_list.append(out_seq)

        print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

        print(np.vstack(out_list).shape)
        # aggregate results

        out_poses = (np.vstack(out_list))[:int(len(audio_raw) * 25 / 16000), :]

        # unnormalize
        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        # std = np.clip(std, a_min=0.01, a_max=None)
        out_poses = np.multiply(out_poses, std) + mean

        # make a BVH
        filename_prefix = transcript_path[:-4].split('/')[-1]
        print(filename_prefix)
        make_bvh(save_path, filename_prefix, out_poses)


def make_bvh(save_path, filename_prefix, poses):
    out_poses = np.pad(poses, ((0, 0), (14, 1)), 'constant', constant_values=(0, 0))
    blendshape_example_path = "... your path/AIWIN/data/训练集/arkit.csv"
    with open(blendshape_example_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            blendshape_example = row
            break
    out_bvh_path = os.path.join(save_path, filename_prefix + '.csv')
    with open(out_bvh_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(blendshape_example)
        writer.writerows(out_poses)


if __name__ == '__main__':
    '''
    python synthesize.py --ckpt_path "... your path/AIWIN/result/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_5_4_onehot/train_multimodal_context/multimodal_context_checkpoint_326.bin" --transcript_path "... your path/AIWIN/data/val/tsv/A10.tsv" --wav_path "... your path/AIWIN/data/val/wav/A10.wav"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--transcript_path", type=str)
    parser.add_argument("--wav_path", type=str)
    args = parser.parse_args()
    main(args.ckpt_path, args.transcript_path, args.wav_path)





