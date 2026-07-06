import os
from os.path import join as pjoin

import numpy as np
import torch

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed


def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_res_model(res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "cce_%s.log" % opt.ext)
    f = open(out_path, 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    # We only need the underlying dataset (mean/std, data_dict with *all* captions
    # per motion id) - not the DataLoader, since CCE needs every caption for a
    # given motion id together in one batch, not one random caption per item.
    _, eval_dataset = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    vq_model.to(opt.device)
    res_model.to(opt.device)

    repeat_time = 20

    for file in sorted(os.listdir(model_dir)):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer = load_trans_model(model_opt, file)
        t2m_transformer.eval()
        t2m_transformer.to(opt.device)

        cce_pose_list = []
        cce_contact_list = []

        for i in range(repeat_time):
            with torch.no_grad():
                cce_pose, cce_contact, num_ids = eval_t2m.evaluate_cross_caption_consistency(
                    eval_dataset, vq_model, res_model, t2m_transformer,
                    time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                    temperature=opt.temperature, topkr=opt.topkr,
                    repeat_id=i, res_cond_scale=5, batch_size=opt.batch_size, device=opt.device)
            cce_pose_list.append(cce_pose)
            cce_contact_list.append(cce_contact)

        cce_pose_arr = np.array(cce_pose_list)
        cce_contact_arr = np.array(cce_contact_list)

        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tCCE-Pose: {np.mean(cce_pose_arr):.4f}, conf. {np.std(cce_pose_arr) * 1.96 / np.sqrt(repeat_time):.4f}\n" \
                    f"\tCCE-Contact: {np.mean(cce_contact_arr):.4f}, conf. {np.std(cce_contact_arr) * 1.96 / np.sqrt(repeat_time):.4f}\n\n"
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()
