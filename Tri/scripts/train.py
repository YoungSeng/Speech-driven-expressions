import pdb
import time
from pathlib import Path
import sys
import pprint

[sys.path.append(i) for i in ['.', '..']]
import torch
import matplotlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_eval.train_gan import train_iter_gan
from utils.average_meter import AverageMeter

matplotlib.use('Agg')  # we don't use interactive GUI

from config.parse_args import parse_args
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils
from myfastdtw import myfastdtw


args = parse_args()
device = torch.device("cuda:" + str(args.no_cuda[0]) if torch.cuda.is_available() else "cpu")

def init_model(args, speaker_model, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = loss_fn = None
    if args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  word_embed_size=args.wordembed_dim,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim, args).to(_device)
    elif args.model == 'speech2gesture':
        loss_fn = torch.nn.L1Loss()

    return generator, discriminator, loss_fn


def one_hot(x):
    x = x.unsqueeze(-1)
    condition = torch.zeros(x.shape[0], 2).scatter_(1, x.type(torch.LongTensor), 1)
    return condition


def train_epochs(args, train_data_loader, test_data_loader, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e10, 0)  # value, epoch, modify

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 10

    # z type
    speaker_model = None

    # init model
    generator, discriminator, loss_fn = init_model(args, speaker_model, pose_dim, device)

    # checkpoint_path = "... your path/AIWIN/result/output_myfastdtw_batchfist_interpolate_normalize_dropout_data_decoder_val3_xlsr/train_multimodal_context/multimodal_context_checkpoint_023.bin"
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # generator.load_state_dict(checkpoint['gen_dict'])
    # generator = generator.to(device)

    # use multi GPUs
    print(torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator, device_ids=[eval(i) for i in args.no_cuda])
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator, device_ids=[eval(i) for i in args.no_cuda])

    # prepare an evaluator for FGD
    embed_space_evaluator = None


    # define optimizers
    decay_rate = 1.0
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=1, gamma=decay_rate)

    # gen_optimizer.load_state_dict(checkpoint['opt'])

    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(0, args.epochs + 0):
        # '''
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation', val_metrics[key], global_iter)
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # '''
        # save model
        # is_best = True
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                'dis_dict': dis_state_dict, 'opt': gen_optimizer.state_dict()
            }, save_name)


        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            _, _, in_text_padded, target_vec, in_audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data
            batch_size = target_vec.size(0)
            in_text_padded = in_text_padded.to(device)

            # noise = torch.randn_like(in_audio) * 0.0003     # modify
            # in_audio = in_audio + noise

            one_hot_embedding_1 = (torch.stack([one_hot(torch.as_tensor([0])) if 'trn_' in i else one_hot(torch.as_tensor([1])) for i in aux_info['vid']])).to(device)
            one_hot_embedding = torch.repeat_interleave(one_hot_embedding_1, repeats=55, dim=1)

            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)

            pitch = pitch.type(torch.FloatTensor).to(device)
            energy = energy.type(torch.FloatTensor).to(device)
            volume = volume.type(torch.FloatTensor).to(device)

            if args.use_emo and args.use_txt_emo:
                text_emo = text_emo.type(torch.FloatTensor).to(device)
            if args.use_emo and args.use_audio_emo:
                speech_emo = speech_emo.type(torch.FloatTensor).to(device)

            # speaker input
            vid_indices = []

            # train
            loss = []
            if args.model == 'multimodal_context':
                loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_vec, vid_indices,
                                      generator, discriminator,
                                      gen_optimizer, dis_optimizer, pitch, energy, volume, speech_emo, text_emo, scheduler, one_hot_embedding)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                           batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))


def evaluate_testset(test_data_loader, generator, loss_fn, embed_space_evaluator, args):
    # to evaluation mode
    generator.train(False)

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            _, _, in_text_padded, target_vec, in_audio, aux_info, pitch, energy, volume, speech_emo, text_emo = data

            batch_size = target_vec.size(0)

            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)

            one_hot_embedding_1 = (torch.stack([one_hot(torch.as_tensor([0])) if 'trn_' in i else one_hot(torch.as_tensor([1])) for i in aux_info['vid']])).to(device)
            one_hot_embedding = torch.repeat_interleave(one_hot_embedding_1, repeats=55, dim=1)

            target = target_vec.to(device)
            pitch = pitch.type(torch.FloatTensor).to(device)
            energy = energy.type(torch.FloatTensor).to(device)
            volume = volume.type(torch.FloatTensor).to(device)
            if args.use_emo and args.use_txt_emo:
                text_emo = text_emo.type(torch.FloatTensor).to(device)
            if args.use_emo and args.use_audio_emo:
                speech_emo = speech_emo.type(torch.FloatTensor).to(device)

            # speaker input
            speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            if args.model == 'multimodal_context':
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices, pitch, energy, volume, speech_emo, text_emo, target, one_hot_embedding)
                # loss = F.l1_loss(out_dir_vec, target)
                # loss += F.mse_loss(out_dir_vec, target)
                # loss = F.mse_loss(out_dir_vec, target)
                out_dir_vec = out_dir_vec.cpu()
                target = target.cpu()
                loss = myfastdtw(out_dir_vec, target)
            else:
                assert False

            losses.update(loss.item(), batch_size)

            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                # calculate MAE of joint coordinates
                out_dir_vec = out_dir_vec.cpu().numpy()
                out_joint_poses = out_dir_vec
                target_vec = target_vec.cpu().numpy()
                target_poses = target_vec
                if out_joint_poses.shape[1] == args.n_poses:
                    diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                else:
                    diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start

    logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s'.format(
        losses.avg, joint_mae.avg, elapsed_time))

    return ret_dict


def main():
    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    train_dataset = TwhDataset(args.train_data_path[0],
                               n_poses=args.n_poses,
                               subdivision_stride=args.subdivision_stride,
                               pose_resampling_fps=args.motion_resampling_framerate,
                               data_mean=args.data_mean, data_std=args.data_std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=word_seq_collate_fn
                              )

    val_dataset = TwhDataset(args.val_data_path[0],
                             n_poses=args.n_poses,
                             subdivision_stride=args.subdivision_stride,
                             pose_resampling_fps=args.motion_resampling_framerate,
                             data_mean=args.data_mean, data_std=args.data_std)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=word_seq_collate_fn
                             )

    print(len(train_loader), len(test_loader))

    # train
    pose_dim = args.pose_dim  # 18 x 3, 27 -> 54
    train_epochs(args, train_loader, test_loader, pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    '''
    python Tri/scripts/train.py --config=/ceph/home/yangsc21/Python/AIWIN/Tri/config/multimodal_context.yml
    '''
    main()
