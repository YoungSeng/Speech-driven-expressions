import torch
import torch.nn as nn

import sys
[sys.path.append(i) for i in ['.', '..']]

from model.tcn import TemporalConvNet
import pdb
from config.parse_args import parse_args
from transformers import BertModel
import torch.nn.functional as F
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

args = parse_args()
device = torch.device("cuda:" + str(args.no_cuda[0]) if torch.cuda.is_available() else "cpu")

if args.use_hubert:
    class WavEncoder(nn.Module):  # (batch, 166, 1024)
        def __init__(self):
            super().__init__()

            model_path = './Tri/chinese-hubert-large/'
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            self.model = HubertModel.from_pretrained(model_path)
            self.model.eval()

            for name, param in self.model.named_parameters():
                param.requires_grad = False

            # self.conv_reduce = nn.Sequential(
            #     nn.Conv1d(1024, 512, 19),
            #     nn.BatchNorm1d(512),
            #     nn.LeakyReLU(inplace=True),
            #     nn.Conv1d(512, 256, 19),
            #     nn.BatchNorm1d(256),
            #     nn.LeakyReLU(inplace=True),
            #     nn.Conv1d(256, 128, 19),
            #     nn.BatchNorm1d(128),
            #     nn.LeakyReLU(inplace=True),
            # )

            self.audio_feature_map = nn.Linear(1024, 64)        # modify

        def forward(self, wav_input_16khz):
            if args.use_myWav2Vec2Model:
                return wav_input_16khz
            else:
                rep = self.feature_extractor(wav_input_16khz, sampling_rate=16000,
                                             return_tensors="pt").input_values.squeeze(0).to(wav_input_16khz.device)
                outputs = self.model(rep).last_hidden_state
                # x = self.conv_reduce(outputs.permute(0, 2, 1))
                # return x.permute(0, 2, 1)
                x = F.interpolate(outputs.transpose(1, 2), size=55, align_corners=True, mode='linear')
                return self.audio_feature_map(x.transpose(1, 2))

else:
    class WavEncoder(nn.Module):  # (batch, 166, 1024)
        def __init__(self):
            super().__init__()
            self.feat_extractor = nn.Sequential(
                nn.Conv1d(1, 16, 10, stride=5, padding=200),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(16, 32, 10, stride=5),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(32, 64, 10, stride=5),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(64, 128, 10, stride=5),
            )

        def forward(self, wav_input_16khz):
            rep = self.feat_extractor(wav_input_16khz.unsqueeze(1))
            return rep.permute(0, 2, 1)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """

    def __init__(self, args, embed_size=768, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.1, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        bert_path = "... your path/AIWIN/chinese-roberta-wwm-ext/"
        self.embedding = BertModel.from_pretrained(bert_path)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input).last_hidden_state)
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class PoseGenerator(nn.Module):
    def __init__(self, args, pose_dim, word_embed_size, z_obj=None):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_obj = z_obj
        self.input_context = args.input_context

        if self.input_context == 'both':
            self.in_size = args.dim_audio + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
        elif self.input_context == 'none':
            self.in_size = pose_dim + 1
        else:
            self.in_size = 32 + pose_dim + 1  # audio or text only

        self.audio_encoder = WavEncoder()
        self.text_encoder = TextEncoderTCN(args, word_embed_size, dropout=args.dropout_prob)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

        self.AIWIN = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64+2, nhead=2, batch_first=True), num_layers=1)
        )
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=64+2, nhead=2, batch_first=True), num_layers=1)
        self.AIWIN_ = nn.Sequential(
            nn.Linear(64+2, 37),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh()
        )

        # self.AIWIN = nn.Sequential(
        #     nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=2, batch_first=True), num_layers=1)
        # )
        # self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=64, nhead=2, batch_first=True), num_layers=1)
        # self.AIWIN_ = nn.Sequential(
        #     nn.Linear(64, 37),
        #     nn.Dropout(p=0.1),  # dropout训练
        #     nn.Tanh()
        # )

        self.args = args

    def forward(self, pre_seq, in_text, in_audio, vid_indices=None, pitch=None, energy=None, volume=None,
                speech_emo=None, text_emo=None, groundtruth=None, one_hot_embedding=None):
        '''
        # print(pre_seq.shape)        # torch.Size([128, 40, 217])
        # print(in_text.shape)        # torch.Size([128, 40])
        # print(in_audio.shape)       # torch.Size([128, 21333])
        '''

        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if self.input_context != 'none':
            # audio
            audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)

            # text
            text_feat_seq, _ = self.text_encoder(in_text)  # torch.Size([128, 40, 32])
            assert (audio_feat_seq.shape[1] == text_feat_seq.shape[1])

        if self.input_context == 'audio':
            # 4.
            # transformer_encoder_output = self.AIWIN(audio_feat_seq)
            # transformer_decoder_output = self.TransformerDecoder(tgt=transformer_encoder_output, memory=audio_feat_seq)
            # output = self.AIWIN_(transformer_decoder_output)
            # 2.
            # in_data = torch.cat((audio_feat_seq, text_feat_seq), dim=2)
            # output = self.AIWIN(in_data)
            # 3.
            # transformer_encoder_output = self.AIWIN(in_data)
            # transformer_decoder_output = self.TransformerDecoder(tgt=transformer_encoder_output, memory=in_data)
            # output = self.AIWIN_(transformer_decoder_output)
            # 5.
            in_data = torch.cat((audio_feat_seq, one_hot_embedding), dim=2)
            transformer_encoder_output = self.AIWIN(in_data)
            transformer_decoder_output = self.TransformerDecoder(tgt=transformer_encoder_output, memory=in_data)
            output = self.AIWIN_(transformer_decoder_output)

        return output, None, None, None, None, None, None


class Discriminator(nn.Module):
    def __init__(self, args, input_size, n_words=None, word_embed_size=None, word_embeddings=None):
        super().__init__()
        self.input_size = input_size

        if n_words and word_embed_size:
            self.text_encoder = TextEncoderTCN(word_embed_size, word_embeddings)
            input_size += 32
        else:
            self.text_encoder = None

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_size, num_layers=args.n_layers, bidirectional=True,
                          dropout=args.dropout_prob, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(args.n_poses, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.text_encoder:
            text_feat_seq, _ = self.text_encoder(in_text)
            poses = torch.cat((poses, text_feat_seq), dim=2)

        output, decoder_hidden = self.gru(poses, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, input_size, args=None):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = nn.GRU(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(49, 1)  # 28 -> 34, to be update!

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        poses = poses.transpose(1, 2)  # torch.Size([128, 27, 34])
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)
        output, decoder_hidden = self.gru(feat, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)
        return output


if __name__ == '__main__':
    '''
    python model/multimodal_context_net.py --config=... your path/AIWIN/Tri/config/multimodal_context.yml
    '''
    pass
