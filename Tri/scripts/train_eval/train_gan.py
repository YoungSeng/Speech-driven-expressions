import pdb

import torch
import torch.nn.functional as F
# from soft_dtw_cuda import SoftDTW
from myfastdtw import myfastdtw

def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


def train_iter_gan(args, epoch, in_text, in_audio, target_poses, vid_indices,
                   pose_decoder, discriminator,
                   pose_dec_optim, dis_optim, pitch, energy, volume, speech_emo, text_emo, scheduler, one_hot_embedding):
    warm_up_epochs = args.loss_warmup
    use_noisy_target = False

    # make pre seq input
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints, 表示限制条件的位

    ###########################################################################################
    # train D
    dis_error = None
    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        dis_optim.zero_grad()

        out_dir_vec, *_ = pose_decoder(pre_seq, in_text, in_audio, vid_indices, pitch, energy, volume, speech_emo, text_emo)  # out shape (batch x seq x dim)

        if use_noisy_target:
            noise_target = add_noise(target_poses)
            noise_out = add_noise(out_dir_vec.detach())
            dis_real = discriminator(noise_target, in_text)
            dis_fake = discriminator(noise_out, in_text)
        else:
            dis_real = discriminator(target_poses, in_text)
            dis_fake = discriminator(out_dir_vec.detach(), in_text)

        dis_error = 0.8 * (torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8))))  # ns-gan
        dis_error.backward()
        torch.nn.utils.clip_grad_norm_(pose_decoder.parameters(), 0.1)  # 规定了最大不能超过的max_norm
        dis_optim.step()

    ###########################################################################################
    # train G
    pose_dec_optim.zero_grad()

    # decoding
    # print(pre_seq.shape)        # torch.Size([128, 40, 217])
    # print(in_text.shape)        # torch.Size([128, 40])
    # print(in_audio.shape)       # torch.Size([128, 21333])
    # print(vid_indices)
    out_dir_vec, z, z_mu, z_logvar, cmd_loss, diff_loss, recon_loss = pose_decoder(pre_seq, in_text, in_audio, vid_indices, pitch, energy, volume, speech_emo, text_emo, target_poses, one_hot_embedding)

    # loss
    # mse_loss = F.mse_loss(out_dir_vec, target_poses)
    # dis_output = discriminator(out_dir_vec, in_text)
    # gen_error = -torch.mean(torch.log(dis_output + 1e-8))
    # loss = mse_loss
    out_dir_vec = out_dir_vec.cpu()
    target_poses = target_poses.cpu()
    loss = myfastdtw(out_dir_vec, target_poses)
    # sdtw = SoftDTW(use_cuda=False, gamma=0.1)
    # sdtw_loss = sdtw(out_dir_vec, target_poses)


    # if args.use_MISA:
    #     if args.use_diff:
    #         loss = 0.9 * (50 * mse_loss + 0.01 * torch.mean(diff_loss) + 0.01 * torch.mean(recon_loss) + 0.01 * torch.mean(cmd_loss))  # + var_loss + cmd_loss + diff_loss + recon_loss
    #     else:
    #         loss = 0.9 * (50 * mse_loss + 0.01 * torch.mean(recon_loss) + 0.01 * torch.mean(cmd_loss))
    # else:
    #     loss = args.loss_regression_weight * mse_loss

    # loss = sdtw_loss.mean()
    loss.backward()

    # if epoch > warm_up_epochs:
    #     loss += args.loss_gan_weight * gen_error

    # pdb.set_trace()
    # loss.backward()
    torch.nn.utils.clip_grad_norm_(pose_decoder.parameters(), 0.1)     # 规定了最大不能超过的max_norm
    pose_dec_optim.step()
    scheduler.step()

    ret_dict = {'loss': loss}

    # if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
    #     ret_dict['gen'] = args.loss_gan_weight * gen_error.item()
    #     ret_dict['dis'] = dis_error.item()
    return ret_dict



