import math
import os
import sys
from os.path import join as pjoin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import codecs as cs

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion_multi
from utils.paramUtil import t2m_kinematic_chain

import numpy as np

clip_version = 'ViT-B/32'
NUM_VIS = 200       # visualize this many motion IDs from test set
MAX_CAPTIONS = 3    # use at most this many captions per motion
VQ_DOWN_FACTOR = 4  # VQ-VAE compresses by factor of 4 (down_t=2, stride_t=2)


def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
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
                      map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, opt, which_model):
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
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_res_model(res_opt, vq_opt, opt):
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


def load_captions(text_path):
    """Return caption strings where f_tag==0 and to_tag==0 (whole-clip captions), up to MAX_CAPTIONS."""
    captions = []
    with cs.open(text_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('#')
            if len(parts) < 4:
                continue
            caption = parts[0]
            try:
                f_tag = float(parts[2])
                to_tag = float(parts[3])
            except ValueError:
                f_tag, to_tag = 0.0, 0.0
            if math.isnan(f_tag):
                f_tag = 0.0
            if math.isnan(to_tag):
                to_tag = 0.0
            if f_tag == 0.0 and to_tag == 0.0:
                captions.append(caption)
            if len(captions) >= MAX_CAPTIONS:
                break
    return captions


if __name__ == '__main__':
    parser = EvalT2MOptions()
    parser.initialize()
    parser.parser.set_defaults(
        gpu_id=0,
        dataset_name='t2m',
        name='t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',
        res_name='tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw',
    )
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263
    nb_joints = 21 if opt.dataset_name == 'kit' else 22

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    result_dir = pjoin('./visualization', opt.ext)
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(animation_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    # ---- dataset paths ----
    if opt.dataset_name == 't2m':
        data_root = './dataset/HumanML3D'
        split_file = pjoin(data_root, 'test.txt')
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        min_motion_len = 40
    else:
        data_root = './dataset/KIT-ML'
        split_file = pjoin(data_root, 'test.txt')
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        min_motion_len = 24

    # ---- load models ----
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    assert res_opt.vq_name == model_opt.vq_name

    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    # ---- mean / std for inv_transform ----
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    def inv_transform(data):
        return data * std + mean

    # ---- build test list (up to NUM_VIS valid entries) ----
    with open(split_file, 'r') as f:
        all_ids = [line.strip() for line in f.readlines()]

    selected = []
    for motion_id in all_ids:
        motion_path = pjoin(motion_dir, motion_id + '.npy')
        text_path = pjoin(text_dir, motion_id + '.txt')
        if not os.path.exists(motion_path) or not os.path.exists(text_path):
            continue
        motion = np.load(motion_path)
        if len(motion) < min_motion_len or len(motion) >= 200:
            continue
        captions = load_captions(text_path)
        if len(captions) == 0:
            continue
        selected.append((motion_id, motion, captions))
        if len(selected) >= NUM_VIS:
            break

    print(f"Selected {len(selected)} motions for visualization.")

    kinematic_chain = t2m_kinematic_chain

    for idx, (motion_id, gt_motion_raw, captions) in enumerate(selected):
        print(f"[{idx+1}/{len(selected)}] {motion_id}  captions={len(captions)}")

        # ---- get token length by actually encoding GT motion through VQ-VAE ----
        with torch.no_grad():
            gt_normalized = (gt_motion_raw - mean) / std
            gt_tensor = torch.from_numpy(gt_normalized).unsqueeze(0).float().to(opt.device)
            code_idx, _ = vq_model.encode(gt_tensor)
            token_len = code_idx.shape[1]

        m_length = token_len * VQ_DOWN_FACTOR
        token_lens = torch.LongTensor([token_len] * len(captions)).to(opt.device)

        # ---- generate motions for all captions ----
        with torch.no_grad():
            mids = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)

        pred_motions = pred_motions.detach().cpu().numpy()
        pred_motions = inv_transform(pred_motions)   # (n_cap, T, dim)

        # ---- GT joints ----
        gt_data = gt_motion_raw[:m_length]
        gt_joint = recover_from_ric(torch.from_numpy(gt_data).float(), nb_joints).numpy()

        # ---- build joints list: [GT, gen_cap1, gen_cap2, ...] ----
        joints_list = [gt_joint]
        for k in range(len(captions)):
            gen_data = pred_motions[k][:m_length]
            gen_joint = recover_from_ric(torch.from_numpy(gen_data).float(), nb_joints).numpy()
            joints_list.append(gen_joint)

        # ---- save combined video ----
        video_path = pjoin(animation_dir, f"{motion_id}.mp4")
        plot_3d_motion_multi(video_path, kinematic_chain, joints_list, fps=20)

        # ---- save caption info ----
        txt_path = pjoin(animation_dir, f"{motion_id}.txt")
        with open(txt_path, 'w') as f:
            for cap_idx, cap in enumerate(captions, 1):
                f.write(f"caption{cap_idx}: {cap}\n")

        print(f"  -> saved {video_path}")
