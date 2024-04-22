import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsvg.difflib.tensor import SVGTensor
from .utils import _get_padding_mask, _get_visibility_mask
from .config import _DefaultConfig


class SVGLoss(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.cfg = cfg

        self.args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

    def forward(self, output, labels, weights):
        loss = 0.
        res = {}

        # VAE
        if self.cfg.use_vae:
            mu, logsigma = output["mu"], output["logsigma"]
            loss_kl = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - torch.exp(logsigma))
            loss_kl = loss_kl.clamp(min=weights["kl_tolerance"])

            loss += weights["loss_kl_weight"] * loss_kl
            res["loss_kl"] = loss_kl

        # remove commitment loss
        # if self.cfg.use_vqvae:
        #     vqvae_loss = output["vqvae_loss"].mean()
        #     loss += vqvae_loss
        #     res["vqvae_loss"] = vqvae_loss

        # Target & predictions
        # tgt_commands.shape [batch_size, max_num_groups, max_seq_len + 2]
        # tgt_args.shape     [batch_size, max_num_groups, max_seq_len + 2, n_args]
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        # 2-stage visibility
        if self.cfg.decode_stages == 2:
            visibility_logits = output["visibility_logits"]
            loss_visibility = F.cross_entropy(visibility_logits.reshape(-1, 2), visibility_mask.reshape(-1).long())

            loss += weights["loss_visibility_weight"] * loss_visibility
            res["loss_visibility"] = loss_visibility

        # Commands & args
        if self.cfg.bin_targets:  # 当使用 bin_targets 时，每个坐标是由 8 bit 代表的，所以会多一维
            tgt_args = tgt_args[..., 1:, :, :]
        else:
            tgt_args = tgt_args[..., 1:, :]
        tgt_commands, padding_mask = tgt_commands[..., 1:], padding_mask[..., 1:]

        # mask.shape [batch_size, 8, 31, 11]
        # 对于预测正确的 command, mask 会乘上 True, cmd_args_mask 向量不会发生改变
        # 对于预测错误的 command, mask 会乘上 False, 相当于把 cmd_args_mask 置为 0, 即不统计对应的 args
        # pred_cmd = torch.argmax(command_logits, dim = -1)
        # mask = self.cmd_args_mask[tgt_commands.long()] * (pred_cmd == tgt_commands).unsqueeze(-1)

        mask = self.cmd_args_mask[tgt_commands.long()]

        
        # padding_mask.shape   [batch_size, num_path, num_commands + 1]
        # command_logits.shape [batch_size, num_path, num_commands + 1, n_commands]
        # command_logits[padding_mask.bool()].shape [-1, n_commands]
        # 目的是把 PAD 的位置筛掉
        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.cfg.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())

        if self.cfg.abs_targets:
            # l2 loss performs better than l1 loss
            loss_args = nn.MSELoss()(
                args_logits[mask.bool()].reshape(-1),
                tgt_args[mask.bool()].reshape(-1).float()
            )
        elif self.cfg.bin_targets:
            loss_args = nn.MSELoss()(
                args_logits[mask.bool()].reshape(-1),
                tgt_args[mask.bool()].reshape(-1).float()
            )
        else:
            loss_args = F.cross_entropy(
                args_logits[mask.bool()].reshape(-1, self.args_dim),
                tgt_args[mask.bool()].reshape(-1).long() + 1
            )  # shift due to -1 PAD_VAL

        loss += weights["loss_cmd_weight"] * loss_cmd \
                + weights["loss_args_weight"] * loss_args

        res.update({
            "loss": loss,
            "loss_cmd": loss_cmd,
            "loss_args": loss_args
        })

        return res
