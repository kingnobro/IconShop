from deepsvg.difflib.tensor import SVGTensor
from deepsvg.utils.utils import _pack_group_batch, _unpack_group_batch, _make_seq_first, _make_batch_first, eval_decorator
from deepsvg.utils import bit2int

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .vector_quantize_pytorch import VectorQuantize
from .basic_blocks import FCN, HierarchFCN, ResNet, ArgumentFCN
from .config import _DefaultConfig
from .utils import (_get_padding_mask, _get_key_padding_mask, _get_group_mask, _get_visibility_mask,
                    _get_key_visibility_mask, _generate_square_subsequent_mask, _sample_categorical, _threshold_sample)

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from scipy.optimize import linear_sum_assignment
from einops import rearrange

from random import randint 


class SVGEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len, use_group=True, group_len=None):
        super().__init__()

        self.cfg = cfg

        # command embedding
        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)  # (7, 256)
        self.embed_fcn = nn.Linear(cfg.n_args, cfg.d_model)

        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len+2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2, dropout=cfg.dropout)

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

        # if not self.cfg.bin_targets:
        #     nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")

        if self.use_group:
            nn.init.kaiming_normal_(self.group_embed.weight, mode="fan_in")

    def forward(self, commands, args, groups=None):
        # commands.shape (32, 960) = (max_seq_len + 2, max_num_groups * batch_size)
        S, GN = commands.shape

        src = self.command_embed(commands.long()) + self.embed_fcn(args)

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)
        return src


class ConstEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len):
        super().__init__()

        self.cfg = cfg

        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len, dropout=cfg.dropout)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.cfg.d_model))
        return src


class LabelEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.label_embedding = nn.Embedding(cfg.n_labels, cfg.dim_label)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.label_embedding.weight, mode="fan_in")

    def forward(self, label):
        src = self.label_embedding(label)
        return src


class Encoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.cfg = cfg

        seq_len = cfg.max_seq_len if cfg.encode_stages == 2 else cfg.max_total_len
        self.use_group = cfg.encode_stages == 1
        self.embedding = SVGEmbedding(cfg, seq_len, use_group=self.use_group)

        if cfg.label_condition:
            self.label_embedding = LabelEmbedding(cfg)
        dim_label = cfg.dim_label if cfg.label_condition else None

        if cfg.model_type == "transformer":
            encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            encoder_norm = LayerNorm(cfg.d_model)
            self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

        else:  # "lstm"
            self.encoder = nn.LSTM(cfg.d_model, cfg.d_model // 2, dropout=cfg.dropout, bidirectional=True)

        if cfg.encode_stages == 2:
            if not cfg.self_match:
                self.hierarchical_PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.max_num_groups)

            # hierarchical_encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            # hierarchical_encoder_norm = LayerNorm(cfg.d_model)
            # self.hierarchical_encoder = TransformerEncoder(hierarchical_encoder_layer, cfg.n_layers, hierarchical_encoder_norm)

    def forward(self, commands, args, label=None):
        # commands.shape: [batch_size, max_num_groups, max_seq_len + 2]
        # args.shape:     [batch_size, max_num_groups, max_seq_len + 2, n_args]
        S, G, N = commands.shape
        l = self.label_embedding(label).unsqueeze(0).unsqueeze(0).repeat(1, commands.size(1), 1, 1) if self.cfg.label_condition else None

        # if self.cfg.encode_stages == 2:
        #     visibility_mask, key_visibility_mask = _get_visibility_mask(commands, seq_dim=0), _get_key_visibility_mask(commands, seq_dim=0)

        commands, args, l = _pack_group_batch(commands, args, l)
        # commands.shape: [batch_size, max_num_groups * (max_seq_len + 2)]
        # key_padding_mask 使得在做 attention 的时候可以遮住 <PAD>
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        # cmd_src, args_src = self.embedding(commands, args, group_mask)
        src = self.embedding(commands, args, group_mask)

        if self.cfg.model_type == "transformer":
            memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask, memory2=l)
            z = memory * padding_mask # 不对 command 做 avg
        else:  # "lstm"
            hidden_cell = (src.new_zeros(2, N, self.cfg.d_model // 2),
                           src.new_zeros(2, N, self.cfg.d_model // 2))
            sequence_lengths = padding_mask.sum(dim=0).squeeze(-1)
            x = pack_padded_sequence(src, sequence_lengths, enforce_sorted=False)

            packed_output, _ = self.encoder(x, hidden_cell)

            memory, _ = pad_packed_sequence(packed_output)
            idx = (sequence_lengths - 1).long().view(1, -1, 1).repeat(1, 1, self.cfg.d_model)
            z = memory.gather(dim=0, index=idx)

        # cmd_z, args_z = _unpack_group_batch(N, cmd_z, args_z)
        z = _unpack_group_batch(N, z)

        # 为什么不用 encode_stages == 1 这个 flag 来实现单个 encoder?
        # 当 encode_stages = 1 时, 获取 data 会有一个 group 操作. 现在尽量不修改原来的代码逻辑
        if self.cfg.one_encoder:
            return z.transpose(0, 1)

        if self.cfg.encode_stages == 2:
            assert False, 'not use E2'
            # src = z.transpose(0, 1)
            # src = _pack_group_batch(src)
            # l = self.label_embedding(label).unsqueeze(0) if self.cfg.label_condition else None

            # if not self.cfg.self_match:
            #     src = self.hierarchical_PE(src)

            # memory = self.hierarchical_encoder(src, mask=None, src_key_padding_mask=key_visibility_mask, memory2=l)

            # if self.cfg.quantize_path:
            #     z = (memory * visibility_mask)
            # else:
            #     z = (memory * visibility_mask).sum(dim=0, keepdim=True) / visibility_mask.sum(dim=0, keepdim=True)
            # z = _unpack_group_batch(N, z)

        return z


class VAE(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(VAE, self).__init__()

        self.enc_mu_fcn = nn.Linear(cfg.d_model, cfg.dim_z)
        self.enc_sigma_fcn = nn.Linear(cfg.d_model, cfg.dim_z)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, z):
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.)
        z = mu + sigma * torch.randn_like(sigma)

        return z, mu, logsigma


class Bottleneck(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Linear(cfg.d_model, cfg.dim_z)

    def forward(self, z):
        return self.bottleneck(z)


class Decoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Decoder, self).__init__()

        self.cfg = cfg

        if cfg.label_condition:
            self.label_embedding = LabelEmbedding(cfg)
        dim_label = cfg.dim_label if cfg.label_condition else None

        if cfg.decode_stages == 2:
            # self.hierarchical_embedding = ConstEmbedding(cfg, cfg.num_groups_proposal)

            # hierarchical_decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            # hierarchical_decoder_norm = LayerNorm(cfg.d_model)
            # self.hierarchical_decoder = TransformerDecoder(hierarchical_decoder_layer, cfg.n_layers_decode, hierarchical_decoder_norm)
            self.hierarchical_fcn = HierarchFCN(cfg.d_model, cfg.dim_z)

        if cfg.pred_mode == "autoregressive":
            self.embedding = SVGEmbedding(cfg, cfg.max_total_len, rel_args=cfg.rel_targets, use_group=True, group_len=cfg.max_total_len)

            square_subsequent_mask = _generate_square_subsequent_mask(self.cfg.max_total_len+1)
            self.register_buffer("square_subsequent_mask", square_subsequent_mask)
        else:  # "one_shot"
            seq_len = cfg.max_seq_len+1 if cfg.decode_stages == 2 else cfg.max_total_len+1
            self.embedding = ConstEmbedding(cfg, seq_len)
            if cfg.args_decoder:
                self.argument_embedding = ConstEmbedding(cfg, seq_len)

        if cfg.model_type == "transformer":
            decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            decoder_norm = LayerNorm(cfg.d_model)
            self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        else:  # "lstm"
            self.fc_hc = nn.Linear(cfg.dim_z, 2 * cfg.d_model)
            self.decoder = nn.LSTM(cfg.d_model, cfg.d_model, dropout=cfg.dropout)

        if cfg.rel_targets:
            args_dim = 2 * cfg.args_dim
        if cfg.bin_targets:
            args_dim = 8
        else:
            args_dim = cfg.args_dim + 1

        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim, cfg.abs_targets)

    def _get_initial_state(self, z):
        hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.cfg.d_model, dim=2)
        hidden_cell = hidden.contiguous(), cell.contiguous()
        return hidden_cell

    def forward(self, z, commands, args, label=None, hierarch_logits=None, return_hierarch=False):
        N = z.size(2)
        l = self.label_embedding(label).unsqueeze(0) if self.cfg.label_condition else None
        if hierarch_logits is None:
            # z = _pack_group_batch(z)
            visibility_z = _pack_group_batch(torch.mean(z[:, 1:, ...], dim=1, keepdim=True))  # 负责预测 visibility, 并且把 SOS 移除

        if self.cfg.decode_stages == 2:
            if hierarch_logits is None:
                # src = self.hierarchical_embedding(z)
                # # print('D2 PE src', src.shape)
                # # print('D2 con z', z.shape)
                # out = self.hierarchical_decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)
                # # print('D2 out', out.shape)
                # hierarch_logits, _z = self.hierarchical_fcn(out)
                # # print('hierarch_logits origin', hierarch_logits.shape)

                # only linear layer for visibility prediction
                hierarch_logits, _z = self.hierarchical_fcn(visibility_z)

            if self.cfg.label_condition: l = l.unsqueeze(0).repeat(1, z.size(1), 1, 1)

            hierarch_logits, l = _pack_group_batch(hierarch_logits, l)
            if not self.cfg.connect_through:
                z = _pack_group_batch(_z)

            if return_hierarch:
                return _unpack_group_batch(N, hierarch_logits, z)

        if self.cfg.pred_mode == "autoregressive":
            S = commands.size(0)
            commands, args = _pack_group_batch(commands, args)

            group_mask = _get_group_mask(commands, seq_dim=0)

            src = self.embedding(commands, args, group_mask)

            if self.cfg.model_type == "transformer":
                key_padding_mask = _get_key_padding_mask(commands, seq_dim=0)
                out = self.decoder(src, z, tgt_mask=self.square_subsequent_mask[:S, :S], tgt_key_padding_mask=key_padding_mask, memory2=l)
            else:  # "lstm"
                hidden_cell = self._get_initial_state(z)  # TODO: reinject intermediate state
                out, _ = self.decoder(src, hidden_cell)

        else:  # "one_shot"
            if self.cfg.connect_through:
                z = rearrange(z, 'p c b d -> c (p b) d')
                z = z[1:, ...]

            src = self.embedding(z)
            out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)
            # print('D1 out', out.shape)

        if self.cfg.args_decoder:
            command_logits = self.command_fcn(out)
            z = torch.argmax(command_logits, dim=-1).unsqueeze(-1).float()
            src = self.argument_embedding(z)
            # print('D0 PE src', src.shape)
            # print('D0 con z', z.shape)
            out = self.argument_decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)
            # print('D0 out', out.shape)
            args_logits = self.argument_fcn(out)
        else:
            # command_logits, args_logits = self.fcn(cmd_out, args_out)
            command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits) + ((hierarch_logits,) if self.cfg.decode_stages == 2 else ())

        return _unpack_group_batch(N, *out_logits)


class SVGTransformer(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(SVGTransformer, self).__init__()

        self.cfg = cfg
        # self.args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1  # 257
        if cfg.rel_targets:
            args_dim = 2 * cfg.args_dim
        if cfg.bin_targets:
            args_dim = 8
        else:
            args_dim = cfg.args_dim + 1

        if self.cfg.encode_stages > 0:

            self.encoder = Encoder(cfg)

            if cfg.use_resnet:
                self.resnet = ResNet(cfg.d_model)

            if cfg.use_vae:
                self.vae = VAE(cfg)
            else:
                self.bottleneck = Bottleneck(cfg)
                # self.bottleneck2 = Bottleneck(cfg)
                self.encoder_norm = LayerNorm(cfg.dim_z, elementwise_affine=False)

            if cfg.use_vqvae:
                self.vqvae = VectorQuantize(
                    dim = cfg.dim_z,
                    codebook_size = cfg.codebook_size,
                    decay = 0.8,
                    commitment_weight = 0.,
                    use_cosine_sim = cfg.use_cosine_sim,
                )

        self.decoder = Decoder(cfg)

        # 定义 self.cmd_args_mask, 但是分配一块持久性缓冲区
        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

    def perfect_matching(self, command_logits, args_logits, hierarch_logits, tgt_commands, tgt_args):
        with torch.no_grad():
            N, G, S, n_args = tgt_args.shape
            visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
            padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

            # Unsqueeze
            tgt_commands, tgt_args, tgt_hierarch = tgt_commands.unsqueeze(2), tgt_args.unsqueeze(2), visibility_mask.unsqueeze(2)
            command_logits, args_logits, hierarch_logits = command_logits.unsqueeze(1), args_logits.unsqueeze(1), hierarch_logits.unsqueeze(1).squeeze(-2)

            # Loss
            tgt_hierarch, hierarch_logits = tgt_hierarch.repeat(1, 1, self.cfg.num_groups_proposal), hierarch_logits.repeat(1, G, 1, 1)
            tgt_commands, command_logits = tgt_commands.repeat(1, 1, self.cfg.num_groups_proposal, 1), command_logits.repeat(1, G, 1, 1, 1)
            tgt_args, args_logits = tgt_args.repeat(1, 1, self.cfg.num_groups_proposal, 1, 1), args_logits.repeat(1, G, 1, 1, 1, 1)

            padding_mask, mask = padding_mask.unsqueeze(2).repeat(1, 1, self.cfg.num_groups_proposal, 1), self.cmd_args_mask[tgt_commands.long()]

            loss_args = F.cross_entropy(args_logits.reshape(-1, self.args_dim), tgt_args.reshape(-1).long() + 1, reduction="none").reshape(N, G, self.cfg.num_groups_proposal, S, n_args)    # shift due to -1 PAD_VAL
            loss_cmd = F.cross_entropy(command_logits.reshape(-1, self.cfg.n_commands), tgt_commands.reshape(-1).long(), reduction="none").reshape(N, G, self.cfg.num_groups_proposal, S)
            loss_hierarch = F.cross_entropy(hierarch_logits.reshape(-1, 2), tgt_hierarch.reshape(-1).long(), reduction="none").reshape(N, G, self.cfg.num_groups_proposal)

            loss_args = (loss_args * mask).sum(dim=[-1, -2]) / mask.sum(dim=[-1, -2])
            loss_cmd = (loss_cmd * padding_mask).sum(dim=-1) / padding_mask.sum(dim=-1)

            loss = 2.0 * loss_args + 1.0 * loss_cmd + 1.0 * loss_hierarch

        # Iterate over the batch-dimension
        assignment_list = []

        full_set = set(range(self.cfg.num_groups_proposal))
        for i in range(N):
            costs = loss[i]
            mask = visibility_mask[i]
            _, assign = linear_sum_assignment(costs[mask].cpu())
            assign = assign.tolist()
            assignment_list.append(assign + list(full_set - set(assign)))

        assignment = torch.tensor(assignment_list, device=command_logits.device)

        return assignment.unsqueeze(-1).unsqueeze(-1)

    @property
    def origin_empty_path(self):
        return torch.tensor([
            11, 16,  7, 23, 24, 10, 13,  5,  1,  8,  3,  3,  7, 15,  7, 18, 15, 31,
            21, 31, 16, 10,  2, 14, 26, 14,  6, 13,  7, 28, 11, 19,  9,  6,  7,  1,
            22, 31, 21,  4, 21,  6,  1,  4, 15, 13, 10, 19,  9, 13, 21, 29, 12, 13,
            10, 23, 15, 11,  1, 18, 19,  5, 23, 20,  7, 29, 13, 15, 22, 31, 17, 10,
            21, 28, 13, 20, 24, 30, 21, 28,  5, 22, 14, 15,  3,  7, 14,  1, 19, 23,
            30, 25, 26, 27, 11, 23,  8,  6,  3, 31, 28, 29, 11,  1,  3,  6,  4, 12,
            12, 25,  0, 18,  5, 26,  5, 12, 23, 14, 19, 25, 12, 20,  2,  3, 18, 11,
            1, 12
        ])

    # for dalle usage
    #   indices = model.get_codebook_indices(*model_args)
    #   commands_y, args_y = model.decode(indices)
    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, commands_enc, args_enc, commands_dec, args_dec):
        indices = self(commands_enc, args_enc, commands_dec, args_dec, return_indices=True)
        return indices

    @torch.no_grad()
    @eval_decorator
    def decode(self, codebook_indices):
        torch.set_printoptions(profile='full')
        print(codebook_indices.reshape(self.cfg.max_num_groups, self.cfg.max_seq_len + 2))
        z = self.vqvae.codebook[codebook_indices]  # shape [batch_size, num_of_indices, codebook_dim]
        # args_z = self.args_vqvae.codebook[codebook_indices]

        batch_size = z.shape[0]
        z = z.reshape(self.cfg.max_num_groups, -1, batch_size, self.cfg.dim_z)

        out_logits = self.decoder(z, None, None)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],  # shape [batch_size, path_num, command_num + 1, 5]
            "args_logits": out_logits[1],     # shape [batch_size, path_num, command_num + 1, 6]
            "visibility_logits": out_logits[2]
        }

        # hack
        # commands_y, args_y, _ = self.greedy_sample(res=res, commands_dec=cmd_indices)
        commands_y, args_y, _ = self.greedy_sample(res=res)

        # visualization, but it is not responsible for decode()
        #   tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
        #   svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().zoom(1.5)
        #   svg_path_sample.fill_(True)
        #   svg_path_sample.save_svg('test.svg')

        return commands_y, args_y

    def forward(self, commands_enc, args_enc, commands_dec, args_dec, label=None,
                z=None, hierarch_logits=None,
                return_tgt=True, params=None, encode_mode=False, return_hierarch=False, return_indices=False):
        # commands_enc 中包含 commands 的类型
        # commands_enc.shape: [batch_size, max_num_groups, max_seq_len + 2]
        # args_enc.shape:     [batch_size, max_num_groups, max_seq_len + 2, n_args]
        # commands_dec.shape: [batch_size, max_num_groups, max_seq_len + 2]
        # args_dec.shape:     [batch_size, max_num_groups, max_seq_len + 2, n_args]
        # assert args_enc.equal(args_dec)
        commands_enc, args_enc = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        commands_dec_, args_dec_ = _make_seq_first(commands_dec, args_dec)
        # commands_enc.shape: [max_seq_len + 2, max_num_groups, batch_size]
        # args_enc.shape:     [max_seq_len + 2, max_num_groups, batch_size, 11]

        if z is None:
            z = self.encoder(commands_enc, args_enc, label)
            # cmd_z, args_z = self.encoder(commands_enc, args_enc, label)
            # print('encoded z', z.shape)

            if self.cfg.use_resnet:
                z = self.resnet(z)

            if self.cfg.use_vae:
                z, mu, logsigma = self.vae(z)
            else:
                # z = self.bottleneck(z)
                z = self.encoder_norm(self.bottleneck(z))
                # cmd_z = self.encoder_norm(self.bottleneck(cmd_z))
                # args_z = self.encoder_norm(self.bottleneck2(args_z))
                # print('bottleneck z', z)
                # print('normed z', z, z.shape)

            if self.cfg.use_vqvae or self.cfg.use_rqvae:
                # initial z.shape [num_path, 1, batch_size, dim_z]
                # batch_size, max_num_groups = cmd_z.shape[2], cmd_z.shape[0]
                batch_size, max_num_groups = z.shape[2], z.shape[0]

                # print(z.shape)
                # z = z.reshape(batch_size, -1, self.cfg.dim_z)
                # z = z.reshape(max_num_groups, -1, self.cfg.dim_z)
                z = rearrange(z, 'p c b z -> b (p c) z')
                # cmd_z = cmd_z.reshape(batch_size, -1, self.cfg.dim_z)
                # args_z = args_z.reshape(batch_size, -1, self.cfg.dim_z)
                # print(z.shape)

                # z = rearrange(z, 'p 1 b d -> b 1 p d')  # p: num_of_path
                #                                         # b: batch_size
                #                                         # d: dim_z
                # z = self.conv_enc_layer(z)
                # z = rearrange(z, 'b c p d -> b (p d) c')      # b d c: batch_size, dim_z, num_channel

                if self.cfg.use_vqvae:
                    quantized, indices, commit_loss = self.vqvae(z) # tokenization
                else:
                    quantized, indices, commit_loss = self.rqvae(z)

                if return_indices:
                    return indices

                # z = rearrange(quantized, 'b (p d) c -> b c p d', p = max_num_groups if self.cfg.quantize_path else 1)
                # z = self.conv_dec_layer(z)
                # z = rearrange(z, 'b 1 p d -> p 1 b d')
                # z = quantized.reshape(max_num_groups, -1, batch_size, self.cfg.dim_z)
                z = rearrange(quantized, 'b (p c) z -> p c b z', p = max_num_groups)

                # cmd_z = cmd_quantized.reshape(max_num_groups, -1, batch_size, self.cfg.dim_z)
                # args_z = args_quantized.reshape(max_num_groups, -1, batch_size, self.cfg.dim_z)
                # print(indices)
                # print('quantized z', z.shape)
        else:
            z = _make_seq_first(z)

        if encode_mode: return z

        if return_tgt:  # Train mode
            # remove EOS command
            # [max_seq_len + 1, max_num_groups, batch_size]
            commands_dec_, args_dec_ = commands_dec_[:-1], args_dec_[:-1]

        out_logits = self.decoder(z, commands_dec_, args_dec_, label, hierarch_logits=hierarch_logits,
                                  return_hierarch=return_hierarch)

        if return_hierarch:
            return out_logits

        out_logits = _make_batch_first(*out_logits)

        if return_tgt and self.cfg.self_match:  # Assignment
            assert self.cfg.decode_stages == 2  # Self-matching expects two-stage decoder
            command_logits, args_logits, hierarch_logits = out_logits

            assignment = self.perfect_matching(command_logits, args_logits, hierarch_logits, commands_dec[..., 1:], args_dec[..., 1:, :])

            command_logits = torch.gather(command_logits, dim=1, index=assignment.expand_as(command_logits))
            args_logits = torch.gather(args_logits, dim=1, index=assignment.unsqueeze(-1).expand_as(args_logits))
            hierarch_logits = torch.gather(hierarch_logits, dim=1, index=assignment.expand_as(hierarch_logits))

            out_logits = (command_logits, args_logits, hierarch_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        if self.cfg.decode_stages == 2:
            res["visibility_logits"] = out_logits[2]

        if return_tgt:
            res["tgt_commands"] = commands_dec
            res["tgt_args"] = args_dec

            if self.cfg.use_vae:
                res["mu"] = _make_batch_first(mu)
                res["logsigma"] = _make_batch_first(logsigma)

            if self.cfg.use_vqvae:
                res["vqvae_loss"] = commit_loss
        return res

    def greedy_sample(self, commands_enc=None, args_enc=None, commands_dec=None, args_dec=None, label=None,
                      z=None, hierarch_logits=None,
                      concat_groups=True, temperature=0.0001, res=None):
        if self.cfg.pred_mode == "one_shot":
            if res is None:
                res = self.forward(commands_enc, args_enc, commands_dec, args_dec, label=label, z=z, hierarch_logits=hierarch_logits, return_tgt=True)

            commands_y = _sample_categorical(temperature, res["command_logits"])
            # hack
            # commands_y = commands_dec.reshape(1, 8, 32)[..., 1:]
            if self.cfg.abs_targets:
                # 此时 args 不需要采样
                # 模型可能直接输出 -1, 所以我们不需要 args_y -= 1
                # 但是 SVG 坐标的范围是 0-255, 我们仍然需要 clamp, 并手动将其转换为整数
                # 那些应该填 "-1" 的位置会在 _make_valid 中被 mask 过滤掉
                # args_y = torch.clamp(res['args_logits'], min=0, max=255).int()
                # args_y = torch.clamp(res['args_logits'], min=0, max=256)
                # args_y = (res['args_logits'] + 1) * 128 - 1
                args_y = (res['args_logits'] + 1) * 12
            elif self.cfg.bin_targets:
                # 此时 args 也不需要采样
                # 我们需要一个 threshold, logits < threshold is 0, logits >= threshold is 1
                threshold = 0.0
                args_logits = res['args_logits']
                args_y = torch.where(args_logits > threshold, torch.ones_like(args_logits), torch.zeros_like(args_logits))
                args_y = bit2int(args_y)
            else:
                args_y = _sample_categorical(temperature, res["args_logits"])
                args_y -= 1  # shift due to -1 PAD_VAL

            visibility_y = _threshold_sample(res["visibility_logits"], threshold=0.7).bool().squeeze(-1) if self.cfg.decode_stages == 2 else None
            commands_y, args_y = self._make_valid(commands_y, args_y, visibility_y)
        else:
            if z is None:
                z = self.forward(commands_enc, args_enc, None, None, label=label, encode_mode=True)

            PAD_VAL = 0
            commands_y, args_y = z.new_zeros(1, 1, 1).fill_(SVGTensor.COMMANDS_SIMPLIFIED.index("SOS")).long(), z.new_ones(1, 1, 1, self.cfg.n_args).fill_(PAD_VAL).long()

            for i in range(self.cfg.max_total_len):
                res = self.forward(None, None, commands_y, args_y, label=label, z=z, hierarch_logits=hierarch_logits, return_tgt=False)
                commands_new_y, args_new_y = _sample_categorical(temperature, res["command_logits"], res["args_logits"])
                args_new_y -= 1  # shift due to -1 PAD_VAL
                _, args_new_y = self._make_valid(commands_new_y, args_new_y)

                commands_y, args_y = torch.cat([commands_y, commands_new_y[..., -1:]], dim=-1), torch.cat([args_y, args_new_y[..., -1:, :]], dim=-2)

            commands_y, args_y = commands_y[..., 1:], args_y[..., 1:, :]  # Discard SOS token

        if self.cfg.rel_targets:
            args_y = self._make_absolute(commands_y, args_y)

        if concat_groups:
            N = commands_y.size(0)
            # 必须使用 commands_y, 而不能用 tgt_commands
            # 因为 commands_y 可能会有多余的 EOS, EOS 是无法可视化的
            padding_mask_y = _get_padding_mask(commands_y, seq_dim=-1).bool()
            commands_y, args_y = commands_y[padding_mask_y].reshape(N, -1), args_y[padding_mask_y].reshape(N, -1, self.cfg.n_args)

        return commands_y, args_y, res

    def _make_valid(self, commands_y, args_y, visibility_y=None, PAD_VAL=0):
        if visibility_y is not None:
            S = commands_y.size(-1)
            commands_y[~visibility_y] = commands_y.new_tensor([SVGTensor.COMMANDS_SIMPLIFIED.index("m"), *[SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")] * (S - 1)])
            args_y[~visibility_y] = PAD_VAL

        mask = self.cmd_args_mask[commands_y.long()].bool()
        args_y[~mask] = PAD_VAL

        return commands_y, args_y

    def _make_absolute(self, commands_y, args_y):

        mask = self.cmd_args_mask[commands_y.long()].bool()
        args_y[mask] -= self.cfg.args_dim - 1

        real_commands = commands_y < SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")

        args_real_commands = args_y[real_commands]
        end_pos = args_real_commands[:-1, SVGTensor.IndexArgs.END_POS].cumsum(dim=0)

        args_real_commands[1:, SVGTensor.IndexArgs.CONTROL1] += end_pos
        args_real_commands[1:, SVGTensor.IndexArgs.CONTROL2] += end_pos
        args_real_commands[1:, SVGTensor.IndexArgs.END_POS] += end_pos

        args_y[real_commands] = args_real_commands

        _, args_y = self._make_valid(commands_y, args_y)

        return args_y
