import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1

        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.opt.device) != "cpu":
            clip.model.convert_weights(
                clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''

        cond = self.mask_cond(cond, force_mask=force_mask)

        # print(motion_ids.shape)
        x = self.token_emb(motion_ids)
        # print(x.shape)
        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)

        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0) #(seqlen+1, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1) #(b, seqlen+1)
        # print(xseq.shape, padding_mask.shape)

        # print(padding_mask.shape, xseq.shape)

        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:] #(seqlen, b, e)
        logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens, counts=None, lambda_consistency=0.0, consistency_type='argmax',
               vq_model=None, all_codes=None, lambda_margin=0.0, confusable_idx=None, margin=1.0,
               code_dist=None, debug_print_topk=False, debug_topk=5, debug_num_motions=3, debug_num_positions=3):
        '''
        :param counts: (b,) 모션별 캡션 개수. None이면 캡션 1개짜리 일반 forward (예: validation).
        :param consistency_type: 'argmax' -> _consistency_option1 (CE 재가중, 기존 방식)
                                 'motion' -> _consistency_option2_motion (soft_emb+STE로 실제 디코드해서 그룹 평균과의 편차)
                                 'motion_lookup' -> _consistency_option3_motion_lookup (soft_emb 없이, 사전계산된
                                 코드간 거리표를 argmax 예측 쌍에 조회해서 CE 재가중)
        :param vq_model, all_codes: consistency_type='motion'일 때 필요한 frozen VQ 디코더/GT 잔차코드
        :param confusable_idx, margin: lambda_margin>0일 때 필요, tools.build_confusable_idx()의 출력
        :param code_dist: consistency_type='motion_lookup'일 때 필요, tools.build_code_distance_matrix()의 출력
        :param debug_print_topk: True면 배치 앞쪽 몇 개 모션의 캡션별 top-k 예측/확률을 stdout에 출력 (디버깅용)
        '''
        bs, ntokens = ids.shape
        device = ids.device

        non_pad_mask = lengths_to_mask(m_lens, ntokens)  # (bs, n)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if counts is not None:
            b = len(counts)
            # 모션별 시작 인덱스 (첫 번째 캡션 행)
            start_indices = torch.zeros(b, dtype=torch.long, device=device)
            if b > 1:
                start_indices[1:] = counts[:-1].cumsum(0)

            ids_b          = ids[start_indices]           # (b, ntokens)
            non_pad_mask_b = non_pad_mask[start_indices]  # (b, ntokens)

            # mask를 모션 단위로 한 번 생성 → K개 캡션에 공유
            rand_time = uniform((b,), device=device)
            rand_mask_probs = self.noise_schedule(rand_time)
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
            batch_randperm = torch.rand((b, ntokens), device=device).argsort(dim=-1)
            mask_b = batch_randperm < num_token_masked.unsqueeze(-1)
            mask_b &= non_pad_mask_b

            # x_ids도 모션 단위로 한 번 생성 → K개 캡션에 공유
            x_ids_b = ids_b.clone()
            mask_rid_b = get_mask_subset_prob(mask_b, 0.1)
            rand_id_b = torch.randint_like(x_ids_b, high=self.opt.num_tokens)
            x_ids_b = torch.where(mask_rid_b, rand_id_b, x_ids_b)
            mask_mid_b = get_mask_subset_prob(mask_b & ~mask_rid_b, 0.88)
            x_ids_b = torch.where(mask_mid_b, self.mask_id, x_ids_b)

            mask  = torch.repeat_interleave(mask_b,   counts, dim=0)   # (bs, ntokens)
            x_ids = torch.repeat_interleave(x_ids_b, counts, dim=0)   # (bs, ntokens)
        else:
            rand_time = uniform((bs,), device=device)
            rand_mask_probs = self.noise_schedule(rand_time)
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
            batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1)
            mask &= non_pad_mask

            x_ids = ids.clone()
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        labels = torch.where(mask, ids, self.mask_id)
        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)

        if counts is not None:
            # per-token CE: (bs, seqlen)
            ce_per_token = F.cross_entropy(logits, labels, ignore_index=self.mask_id, reduction='none')
            # per-sample mean (토큰 균등)
            ce_per_sample = (ce_per_token * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
            # per-motion mean (캡션 간 균등)
            motion_idx = torch.repeat_interleave(torch.arange(b, device=device), counts)
            motion_ce = torch.zeros(b, device=device).scatter_add(0, motion_idx, ce_per_sample) / counts.float()
            ce_loss = motion_ce.mean()

            pred_id = logits.argmax(dim=1)
            acc = (pred_id == labels).masked_select(mask).float().mean().item()

            if debug_print_topk:
                print_caption_topk_debug(logits, labels, ids, mask, counts,
                                         topk=debug_topk, num_motions=debug_num_motions,
                                         num_positions=debug_num_positions)

            total_loss = ce_loss
            log_dict = {'ce_loss': ce_loss.item()}

            if lambda_consistency > 0:
                if consistency_type == 'argmax':
                    cons_loss = self._consistency_option1(logits, labels, ids, mask, counts)
                elif consistency_type == 'motion':
                    assert vq_model is not None and all_codes is not None, \
                        "consistency_type='motion' requires vq_model and all_codes"
                    cons_loss = self._consistency_option2_motion(logits, mask, all_codes, counts, vq_model)
                elif consistency_type == 'motion_lookup':
                    assert code_dist is not None, \
                        "consistency_type='motion_lookup' requires code_dist"
                    cons_loss = self._consistency_option3_motion_lookup(logits, labels, mask, counts, code_dist)
                else:
                    raise NotImplementedError(f"Unknown consistency_type: {consistency_type}")
                total_loss = total_loss + lambda_consistency * cons_loss
                log_dict['cons_loss'] = cons_loss.item()

            if lambda_margin > 0:
                assert confusable_idx is not None, "lambda_margin>0 requires confusable_idx"
                margin_loss = confusable_margin_loss(logits, labels, mask, confusable_idx, margin=margin)
                total_loss = total_loss + lambda_margin * margin_loss
                log_dict['margin_loss'] = margin_loss.item()

            return total_loss, pred_id, acc, log_dict
        else:
            ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
            return ce_loss, pred_id, acc, {'ce_loss': ce_loss.item()}

    def _consistency_option2_motion(self, logits, mask, all_codes, counts, vq_model):
        '''
        같은 모션의 K개 캡션이 예측한 토큰을 각각 raw motion으로 디코드한 뒤,
        (GT와 비교하는 게 아니라) 그룹 평균 모션 대비 얼마나 벗어나는지를 loss로 준다.
        예측 토큰이 서로 달라도 디코드된 raw motion이 비슷하면 loss가 작다 — RVQ의
        code aliasing(비슷하게 디코드되는 서로 다른 코드)을 자연스럽게 흡수하기 위함.
        마스크 안 된(=컨텍스트로 그대로 넣어준) 위치는 캡션마다 이미 동일한 GT라
        비교에서 제외한다 (frame_mask).
        :param logits: (bs, K, n)
        :param mask: (bs, n) bool, 모션 단위로 공유되는 mask (그룹 내 모든 캡션이 동일)
        :param all_codes: (q, bs, code_dim, n), GT residual-VQ 코드. ids와 동일하게 캡션 수만큼 확장되어 있어야 함
        :param counts: (b,) 모션별 캡션 개수
        :param vq_model: frozen RVQVAE
        '''
        codebook0 = vq_model.quantizer.codebooks[0]  # (K, code_dim)

        probs = F.softmax(logits, dim=1)                            # (bs, K, n)
        z_soft = torch.einsum('bkn,kd->bdn', probs, codebook0)       # (bs, code_dim, n)
        hard_idx = probs.argmax(dim=1)                                # (bs, n)
        z_hard = F.embedding(hard_idx, codebook0).permute(0, 2, 1)    # (bs, code_dim, n)
        # Straight-Through: forward는 실제 codebook 벡터, backward는 soft 분포로 흐름
        z_st = z_hard.detach() + (z_soft - z_soft.detach())

        mask_c = mask.unsqueeze(1)                                    # (bs, 1, n)
        layer0 = torch.where(mask_c, z_st, all_codes[0])              # 컨텍스트는 GT 유지, 마스크 위치만 예측값
        z_full = layer0 + all_codes[1:].sum(dim=0)                    # 잔차 레이어는 GT로 teacher-force

        x_hat = vq_model.decoder(z_full)                              # (bs, T, dim_pose)

        unit_length = self.opt.unit_length
        frame_mask = mask.unsqueeze(-1).expand(-1, -1, unit_length).reshape(mask.shape[0], -1)  # (bs, n*unit)
        frame_mask = frame_mask[:, :x_hat.shape[1]].float().unsqueeze(-1)                        # (bs, T, 1)

        loss = x_hat.new_zeros(())
        n_groups = 0
        start = 0
        for k in counts:
            k = k.item()
            if k > 1:
                group = x_hat[start:start + k]              # (k, T, d)
                group_mask = frame_mask[start:start + k]     # (k, T, 1), 그룹 내 전부 동일
                denom = group_mask.sum(dim=0, keepdim=True).clamp(min=1)
                group_mean = (group * group_mask).sum(dim=0, keepdim=True) / denom  # (1, T, d)
                diff = ((group - group_mean).abs() * group_mask).sum()
                n_valid = (group_mask.sum() * group.shape[-1]).clamp(min=1)
                loss = loss + diff / n_valid
                n_groups += 1
            start += k

        return loss / max(n_groups, 1)

    def _consistency_option3_motion_lookup(self, logits, labels, mask, counts, code_dist):
        '''
        _consistency_option2_motion과 목적은 같지만(캡션끼리 raw-motion이 일치하도록),
        soft_emb/decoder를 전혀 태우지 않는다. 대신 사전계산된 코드간 raw-motion 거리표
        (code_dist, K×K, 0~1 정규화)를 argmax 예측 쌍에 조회해서, CE를 재가중하는 방식.
        _consistency_option1과 뼈대는 동일하고(hard argmax + no_grad + CE 재가중),
        "맞았다/틀렸다" 이진 가중치 대신 raw-motion 거리라는 연속값을 가중치로 쓴다는 점만 다르다.
        :param logits: (bs, K, n)
        :param labels: (bs, n)
        :param mask: (bs, n) bool, 모션 단위로 공유되는 mask
        :param counts: (b,) 모션별 캡션 개수
        :param code_dist: (K, K), tools.build_code_distance_matrix()의 출력
        '''
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (bs, n), hard, soft_emb 없음
            cons_weight = torch.zeros_like(mask, dtype=torch.float)

            start = 0
            for k in counts:
                k = k.item()
                masked = mask[start]  # (n,), 그룹 내 공유

                if k >= 2:
                    preds_i = preds[start:start + k]  # (k, n)
                    pair_dist = torch.zeros_like(masked, dtype=torch.float)
                    n_pairs = 0
                    for a in range(k):
                        for b in range(a + 1, k):
                            # 캡션 a, b가 예측한 토큰이 raw-motion 상 얼마나 다른지 조회
                            pair_dist = pair_dist + code_dist[preds_i[a], preds_i[b]]
                            n_pairs += 1
                    weight_i = (pair_dist / n_pairs) * masked.float()
                else:
                    weight_i = torch.zeros_like(masked, dtype=torch.float)

                cons_weight[start:start + k] = weight_i.unsqueeze(0).expand(k, -1)
                start += k

        ce_per_token = F.cross_entropy(logits, labels, ignore_index=self.mask_id, reduction='none')
        n = cons_weight.sum().clamp(min=1)
        return (ce_per_token * cons_weight).sum() / n

    def _consistency_option1(self, logits, labels, ids, mask, counts):
        '''
        조건1: 모든 캡션이 해당 위치를 GT와 다르게 예측 → +1 가중치
        조건2: 3개 이상의 캡션이 해당 위치에서 서로 다른 값을 예측 → +1 가중치
        두 조건은 독립적으로 합산 (둘 다 만족하면 +2)
        '''
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (b*K_total, seqlen)
            cons_weight = torch.zeros_like(mask, dtype=torch.float)

            start = 0
            for k in counts:
                k = k.item()
                preds_i  = preds[start:start+k]  # (K_i, seqlen)
                gt_i     = ids[start]             # (seqlen)
                masked   = mask[start]            # (seqlen) — 공유 mask이므로 첫 행과 동일

                # 조건1: 모든 캡션이 GT와 다르게 예측
                all_wrong = (preds_i != gt_i.unsqueeze(0)).all(dim=0)  # (seqlen)

                # 조건2: 3개 이상의 서로 다른 값 예측
                if k >= 3:
                    sorted_preds = preds_i.sort(dim=0).values                        # (K_i, seqlen)
                    unique_count = (sorted_preds[1:] != sorted_preds[:-1]).sum(dim=0) + 1  # (seqlen)
                    diverse = unique_count >= 3
                else:
                    diverse = torch.zeros(preds_i.shape[1], dtype=torch.bool, device=preds_i.device)

                # all_wrong → +1, all_wrong & diverse → +1 more (총 +2)
                weight_i = (all_wrong.float() + (all_wrong & diverse).float()) * masked.float()
                cons_weight[start:start+k] = weight_i.unsqueeze(0).expand(k, -1)

                start += k

        ce_per_token = F.cross_entropy(logits, labels, ignore_index=self.mask_id, reduction='none')
        n = cons_weight.sum().clamp(min=1)
        return (ce_per_token * cons_weight).sum() / n
        
    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids


    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             tokens,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             ):

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        if edit_mask == None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self,
                  conds,
                  conds_og,
                  tokens,
                  m_lens,
                  cond_scale: int,
                  force_mask=False,
                  ):

        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
                if conds_og is not None:
                    cond_vector_og = self.encode_text(conds_og)
                else:
                    cond_vector_og = None
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
            if conds_og is not None:
                cond_vector_og = self.enc_action(conds_og).to(device)
            else:
                cond_vector_og = None
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              cond_vector_neg=cond_vector_og,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
        og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

        return og_tokens_scores


class ResidualTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1,
                 num_heads=4, dropout=0.1, clip_dim=512, shared_codebook=False, share_weight=False,
                 clip_version=None, opt=None, **kargs):
        super(ResidualTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        # assert shared_codebook == True, "Only support shared codebook right now!"

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        # self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        self.cond_drop_prob = cond_drop_prob

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 1  # one dummy tokens for padding
        self.pad_id = opt.num_tokens

        # self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers-1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                # self.output_proj_bias = 0
                self.output_proj_weight = output_proj.expand(opt.num_quantizers-1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers-1, _num_tokens)

        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                output_proj_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))

                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                token_embed_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    # def

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.opt.device) != "cpu":
            clip.model.convert_weights(
                clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text


    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            # if not self.registered:
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)
                # self.registered = True

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)

        :return:
            -logits (bs, ntoken, seqlen)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output



    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False):
        '''
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(motion_codes)

        # (b, num_quantizer)
        q_onehot = self.encode_quant(qids).float().to(x.device)

        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, q_emb, x], dim=0)  # (seqlen+2, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, seqlen+2)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[2:]  # (seqlen, b, e)
        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        bs = motion_codes.shape[0]
        # if cond_scale == 1:
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
            logits = self.output_project(logits, qids-1)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask)
        logits = self.output_project(logits, qids-1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids-1)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def forward(self, all_indices, y, m_lens):
        '''
        :param all_indices: (b, n, q)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens)  # (b, n)

        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id) #(b, n, q)

        # randomly sample quantization layers to work on, [1, num_q)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        # print(self.token_embed_weight.shape, all_indices.shape)
        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        # print(token_embed.shape, gather_indices.shape)
        all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)

        cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
        logits = self.output_project(logits, active_q_layers-1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 conds,
                 m_lens,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1, # If it's -1, use all.
                 ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

        for i in range(1, num_quant_layers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        # all_indices = all_indices.masked_fill()
        return all_indices

    @torch.no_grad()
    @eval_decorator
    def edit(self,
            motion_ids,
            conds,
            m_lens,
            temperature=1,
            topk_filter_thres=0.9,
            cond_scale=2
            ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        # all_indices = all_indices.masked_fill()
        return all_indices