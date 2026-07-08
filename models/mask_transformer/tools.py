import torch
import torch.nn.functional as F
import math
from einops import rearrange

# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids==special_id)
    return mask

# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# classifier free guidance functions

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

# noise schedules

# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low

def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


@torch.no_grad()
def print_caption_topk_debug(logits, labels, ids, mask, counts, topk=5, num_motions=3, num_positions=3):
    '''
    배치 앞쪽 num_motions개 모션에 대해, 마스킹된 위치 num_positions개씩만 골라
    캡션별 top-k 예측 토큰/확률을 출력한다. GT와 일치하는 토큰에는 * 표시.
    예: cap0: [240*-0.32, 168-0.21, 236-0.15, 186-0.09, 114-0.05]
    :param logits: (bs, K, n)
    :param labels: (bs, n) — 사용 안 함(호환용으로만 받음), GT는 ids에서 가져옴
    :param ids: (bs, n), 마스킹 전 원본 GT 토큰
    :param mask: (bs, n) bool, 모션 단위로 공유되는 mask
    :param counts: (b,) 모션별 캡션 개수
    '''
    probs = F.softmax(logits, dim=1)                        # (bs, K, n)
    topk_probs, topk_idx = probs.topk(topk, dim=1)          # (bs, topk, n)

    start = 0
    for m_idx, k in enumerate(counts):
        k = k.item()
        if m_idx >= num_motions:
            break

        gt_i = ids[start]        # (n,), 그룹 내 GT는 캡션과 무관하게 동일
        mask_i = mask[start]     # (n,), 그룹 내 mask도 공유

        pos_list = mask_i.nonzero(as_tuple=True)[0][:num_positions].tolist()

        print(f"Motion {m_idx}  ({k} captions)")
        for p_i, pos in enumerate(pos_list):
            gt = gt_i[pos].item()
            print(f"  [masked pos {p_i + 1}={pos}  GT={gt}]")
            for c_i in range(k):
                toks = topk_idx[start + c_i, :, pos].tolist()
                ps = topk_probs[start + c_i, :, pos].tolist()
                items = [f"{t}{'*' if t == gt else ''}-{p:.2f}" for t, p in zip(toks, ps)]
                print(f"    cap{c_i}: [{', '.join(items)}]")

            if k >= 2:
                overlaps = []
                for a in range(k):
                    set_a = set(topk_idx[start + a, :, pos].tolist())
                    for b in range(a + 1, k):
                        set_b = set(topk_idx[start + b, :, pos].tolist())
                        overlaps.append(len(set_a & set_b))
                avg_overlap = sum(overlaps) / len(overlaps)
                print(f"    avg pairwise top{topk} overlap: {avg_overlap:.2f}/{topk}")

        start += k


@torch.no_grad()
def build_confusable_idx(vq_model, topk=5, device=None):
    '''
    Base-layer 코드북의 K개 코드를 각각 단독으로(잔차 없이) 디코드해서
    raw-motion 공간에서 서로 가장 가까운 topk 이웃 코드를 찾아둔다.
    vq_model은 frozen이므로 학습 시작 전 딱 한 번만 호출하면 된다.
    :param vq_model: frozen RVQVAE
    :param topk: 코드별로 남겨둘 confusable neighbor 개수
    :return: (K, topk) long tensor, confusable_idx[g] = GT가 g일 때 헷갈리는 코드 인덱스들
    '''
    codebook0 = vq_model.quantizer.codebooks[0]  # (K, code_dim)
    z = codebook0.unsqueeze(-1)                  # (K, code_dim, 1), 단일 프레임, 잔차 없음
    decoded = vq_model.decoder(z)                # (K, unit_length, dim_pose)
    flat = decoded.reshape(decoded.shape[0], -1)
    dist = torch.cdist(flat, flat)               # (K, K)
    dist.fill_diagonal_(float('inf'))            # 자기 자신은 제외
    confusable_idx = dist.topk(topk, dim=-1, largest=False).indices  # (K, topk)
    return confusable_idx.to(device) if device is not None else confusable_idx


@torch.no_grad()
def build_code_distance_matrix(vq_model, device=None, scale=2.0):
    '''
    Base-layer 코드북의 K개 코드를 각각 단독으로(잔차 없이) 디코드해서
    raw-motion 공간에서의 전체 K×K 거리행렬을 만들어둔다 (0~scale로 정규화).
    build_confusable_idx와 달리 top-k만 남기지 않고 전체를 보존 — 임의의 두 코드
    쌍(예: 서로 다른 캡션이 예측한 토큰)이 얼마나 다른지 조회하는 용도.
    vq_model은 frozen이므로 학습 시작 전 딱 한 번만 호출하면 된다.
    :param scale: 정규화 후 최댓값 (기본 2.0 -> 0~2 범위)
    :return: (K, K) float tensor, 0=완전히 같게 디코드됨 ~ scale=가장 멀리 디코드됨
    '''
    codebook0 = vq_model.quantizer.codebooks[0]  # (K, code_dim)
    z = codebook0.unsqueeze(-1)                  # (K, code_dim, 1), 단일 프레임, 잔차 없음
    decoded = vq_model.decoder(z)                # (K, unit_length, dim_pose)
    flat = decoded.reshape(decoded.shape[0], -1)
    dist = torch.cdist(flat, flat)               # (K, K)
    dist = dist / dist.max().clamp(min=1e-6) * scale  # CE 가중치로 쓸 것이므로 스케일을 0~scale로 고정
    return dist.to(device) if device is not None else dist


def confusable_margin_loss(logits, labels, mask, confusable_idx, margin=1.0):
    '''
    GT 코드와 raw-motion 상 가까운(=헷갈리는) 코드들에 대해서만,
    GT 로짓과의 격차가 margin 이상 벌어지도록 강제하는 hinge loss.
    CE와 달리 "멀리 떨어진 오답"은 건드리지 않고, confusable_idx로 미리 골라둔
    "비슷해서 헷갈리는 오답"에 대해서만 추가로 경계를 세게 그어준다.
    :param logits: (bs, K, n)
    :param labels: (bs, n), 마스크 안 된 위치는 mask_id 등 K 범위 밖 값일 수 있음
    :param mask: (bs, n) bool, True인 위치만 loss에 반영
    :param confusable_idx: (K, topk) long, build_confusable_idx()의 출력
    '''
    logits_p = logits.permute(0, 2, 1)  # (bs, n, K)
    # mask_id 등 K 범위 밖 라벨이 인덱싱에 쓰이지 않도록 안전한 값(0)으로 치환
    gt_safe = torch.where(mask, labels, torch.zeros_like(labels))

    neighbors = confusable_idx[gt_safe]                          # (bs, n, topk)
    logit_g = torch.gather(logits_p, 2, gt_safe.unsqueeze(-1))   # (bs, n, 1)
    logit_k = torch.gather(logits_p, 2, neighbors)               # (bs, n, topk)

    violation = (logit_k - logit_g + margin).clamp(min=0)        # (bs, n, topk)
    valid = mask.float().unsqueeze(-1)                           # (bs, n, 1)

    return (violation * valid).sum() / (valid.sum() * neighbors.shape[-1]).clamp(min=1)


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss