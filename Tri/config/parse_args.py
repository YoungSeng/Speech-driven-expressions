import configargparse
# from pathlib import Path

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


import torch.nn as nn
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh, 'my': nn.LeakyReLU(inplace=True)}


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=False, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main")
    parser.add("--train_data_path", action="append")
    parser.add("--val_data_path", action="append")
    parser.add("--test_data_path", action="append")
    parser.add("--model_save_path", required=False)
    parser.add("--pose_representation", type=str, default='3d_vec')
    parser.add("--data_mean", action="append", type=float, nargs='*')
    parser.add("--data_std", action="append", type=float, nargs='*')
    parser.add("--random_seed", type=int, default=-1)
    parser.add("--save_result_video", type=str2bool, default=True)

    # word embedding
    parser.add("--wordembed_dim", type=int, default=100)
    parser.add("--freeze_wordembed", type=str2bool, default=False)

    # model
    parser.add("--model", type=str, default='multimodal_context')
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--n_layers", type=int, default=2)
    parser.add("--hidden_size", type=int, default=200)
    parser.add("--z_type", type=str, default='none')
    parser.add("--input_context", type=str, default='audio')

    # dataset
    parser.add("--motion_resampling_framerate", type=int, default=25)
    parser.add("--n_poses", type=int, default=55)
    parser.add("--n_pre_poses", type=int, default=0)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=0)

    # GAN parameter
    parser.add("--GAN_noise_size", type=int, default=0)

    # training
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--discriminator_lr_weight", type=float, default=0.2)
    parser.add("--loss_regression_weight", type=float, default=50)
    parser.add("--loss_gan_weight", type=float, default=1.0)
    parser.add("--loss_kld_weight", type=float, default=0.1)
    parser.add("--loss_reg_weight", type=float, default=0.01)
    parser.add("--loss_warmup", type=int, default=-1)
    parser.add("--dim_audio", type=int, default=128)
    parser.add("--dim_text", type=int, default=32)
    parser.add("--dim_video", type=int, default=37)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', default=activation_dict['my'])
    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)       # 20220621
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    parser.add_argument('--size_space', type=int, default=48)       # 20220701 128 -> 32
    parser.add_argument('--pose_dim', type=int, default=36)
    parser.add("--no_cuda", type=list, default=['3','0','1','2'])
    parser.add_argument('--use_emo', type=str2bool, default=False)
    parser.add_argument('--use_hubert', type=str2bool, default=True)
    parser.add_argument('--use_txt_emo', type=str2bool, default=False)
    parser.add_argument('--use_audio_emo', type=str2bool, default=False)
    parser.add_argument('--ckpt_path', type=str, default="... your path/Tri/output_2/train_multimodal_context/multimodal_context_checkpoint_010.bin")  # Not use, just define
    parser.add_argument('--transcript_path', type=str, default="... your path/dataset/v1_18/val/tsv/val_2022_v1_000.tsv")  # Not use, just define
    parser.add_argument('--wav_path', type=str, default="... your path/dataset/v1_18/val/wav/val_2022_v1_000.wav")  # Not use, just define
    parser.add_argument('--use_MISA', type=str2bool, default=False)
    parser.add_argument('--use_diff', type=str2bool, default=False)
    parser.add_argument('--use_faceformer', type=str2bool, default=False)
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=37 * 1, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--use_myWav2Vec2Model", type=str2bool, default=False)

    args = parser.parse_args()
    return args
