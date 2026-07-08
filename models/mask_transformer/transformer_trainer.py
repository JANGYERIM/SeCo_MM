import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        # confusable pair margin loss: 코드북이 frozen이므로 학습 시작 전 딱 한 번만 계산해두면 됨
        self.confusable_idx = None
        if getattr(args, 'lambda_margin', 0.0) > 0:
            self.confusable_idx = build_confusable_idx(
                self.vq_model, topk=args.margin_topk, device=self.device)

        # motion_lookup consistency: 코드 간 raw-motion 거리표도 마찬가지로 한 번만 계산
        self.code_dist = None
        if getattr(args, 'consistency_type', None) == 'motion_lookup':
            self.code_dist = build_code_distance_matrix(self.vq_model, device=self.device)

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):
        captions_list, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        # validation: 샘플당 캡션 1개만 사용
        conds = [caps[0] for caps in captions_list]

        _loss, _pred_ids, _acc, _log = self.t2m_transformer(code_idx[..., 0], conds, m_lens)

        return _loss, _acc

    def update(self, batch_data, it=None):
        captions_list, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # all_codes: (q, b, code_dim, n), consistency_type='motion'일 때만 실제로 쓰이는 GT 잔차코드
        # (margin loss는 confusable_idx만 있으면 되고 all_codes/vq_model은 필요 없음)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens_vq = m_lens // 4
        ids = code_idx[..., 0]  # (b, seqlen)

        # 배치 확장: 각 샘플의 모든 캡션을 독립 샘플로
        counts = torch.tensor([len(caps) for caps in captions_list], device=self.device)
        flat_captions = [cap for caps in captions_list for cap in caps]
        ids_expanded = torch.repeat_interleave(ids, counts, dim=0)
        m_lens_expanded = torch.repeat_interleave(m_lens_vq, counts, dim=0)

        extra_kwargs = {}
        if self.opt.consistency_type == 'motion':
            # all_codes 배치 축(dim=1)도 캡션 수만큼 함께 확장 (motion consistency에서만 필요)
            all_codes_expanded = torch.repeat_interleave(all_codes, counts, dim=1)
            extra_kwargs['vq_model'] = self.vq_model
            extra_kwargs['all_codes'] = all_codes_expanded
        elif self.opt.lambda_margin > 0:
            # margin loss는 vq_model.decoder를 태우지 않고 confusable_idx(로짓 인덱싱)만 쓰므로
            # vq_model/all_codes 없이도 동작한다
            pass

        # "p/it:.. loss:.." 요약 라인이 찍히는 스텝(it % log_every == 0)에만 debug 로그도 같이 출력
        debug_print_topk = self.opt.debug_print_topk
        if debug_print_topk and it is not None:
            debug_print_topk = (it % self.opt.log_every == 0)

        loss, _pred_ids, acc, log_dict = self.t2m_transformer(
            ids_expanded, flat_captions, m_lens_expanded,
            counts=counts,
            lambda_consistency=self.opt.lambda_consistency,
            consistency_type=self.opt.consistency_type,
            lambda_margin=self.opt.lambda_margin,
            confusable_idx=self.confusable_idx,
            margin=self.opt.margin,
            code_dist=self.code_dist,
            debug_print_topk=debug_print_topk,
            debug_topk=self.opt.debug_topk,
            debug_num_motions=self.opt.debug_num_motions,
            debug_num_positions=self.opt.debug_num_positions,
            **extra_kwargs,
        )

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc, log_dict

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc, log_dict = self.update(batch_data=batch, it=it)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']
                for tag, value in log_dict.items():
                    logs[tag] += value

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            # self.l1_criterion = torch.nn.SmoothL1Loss()


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
            self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_loss = 100
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                print(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )