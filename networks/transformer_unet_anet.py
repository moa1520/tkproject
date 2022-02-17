import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from networks.transformer_layers import GraphSelfAttention, GraphEncoderDecoderAttention


class Transformer(nn.Module):

    def __init__(self, nqueries, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder1 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder2 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder3 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder4 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder5 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder6 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder1 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.decoder2 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.decoder3 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.decoder4 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.decoder5 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.decoder6 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, query_embed, pos_embed):
        bs, c, t = src.shape
        src = src.permute(2, 0, 1)
        src1 = src[:96, :, :]
        src2 = src[96:96+48, :, :]
        src3 = src[96+48:96+48+24, :, :]
        src4 = src[96+48+24:96+48+24+12, :, :]
        src5 = src[96+48+24+12:96+48+24+12+6, :, :]
        src6 = src[96+48+24+12+6:, :, :]

        pos_embed = pos_embed.permute(2, 0, 1)
        pos_embed1 = pos_embed[:96, :, :]
        pos_embed2 = pos_embed[96:96+48, :, :]
        pos_embed3 = pos_embed[96+48:96+48+24, :, :]
        pos_embed4 = pos_embed[96+48+24:96+48+24+12, :, :]
        pos_embed5 = pos_embed[96+48+24+12:96+48+24+12+6, :, :]
        pos_embed6 = pos_embed[96+48+24+12+6:, :, :]

        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        query_embed1 = query_embed[:96, :, :]
        query_embed2 = query_embed[96:96+48, :, :]
        query_embed3 = query_embed[96+48:96+48+24, :, :]
        query_embed4 = query_embed[96+48+24:96+48+24+12, :, :]
        query_embed5 = query_embed[96+48+24+12:96+48+24+12+6, :, :]
        query_embed6 = query_embed[96+48+24+12+6:, :, :]

        src_mask1 = src_mask[:, :96]
        src_mask2 = src_mask[:, 96:96+48]
        src_mask3 = src_mask[:, 96+48:96+48+24]
        src_mask4 = src_mask[:, 96+48+24:96+48+24+12]
        src_mask5 = src_mask[:, 96+48+24+12:96+48+24+12+6]
        src_mask6 = src_mask[:, 96+48+24+12+6:]
        encoder_mask = None

        memory1 = self.encoder1(src1, mask=encoder_mask,
                                src_key_padding_mask=src_mask1, pos=pos_embed1)
        memory2 = self.encoder2(src2, mask=encoder_mask,
                                src_key_padding_mask=src_mask2, pos=pos_embed2)
        memory3 = self.encoder3(src3, mask=encoder_mask,
                                src_key_padding_mask=src_mask3, pos=pos_embed3)
        memory4 = self.encoder4(src4, mask=encoder_mask,
                                src_key_padding_mask=src_mask4, pos=pos_embed4)
        memory5 = self.encoder5(src5, mask=encoder_mask,
                                src_key_padding_mask=src_mask5, pos=pos_embed5)
        memory6 = self.encoder6(src6, mask=encoder_mask,
                                src_key_padding_mask=src_mask6, pos=pos_embed6)

        ctx1 = memory1
        ctx2 = memory2
        ctx3 = memory3
        ctx4 = memory4
        ctx5 = memory5
        ctx6 = memory6

        tgt_mask = None
        tgt6 = torch.zeros_like(query_embed6)
        hs6 = self.decoder6(tgt6, ctx6, tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_mask6, pos=pos_embed6, query_pos=query_embed6)
        query_embed5 = torch.cat([query_embed5, hs6[-1]], dim=0)

        tgt5 = torch.zeros_like(query_embed5)
        hs5 = self.decoder5(tgt5, ctx5, tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_mask5, pos=pos_embed5, query_pos=query_embed5)
        query_embed4 = torch.cat([query_embed4, hs5[-1]], dim=0)

        tgt4 = torch.zeros_like(query_embed4)
        hs4 = self.decoder4(tgt4, ctx4, tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_mask4, pos=pos_embed4, query_pos=query_embed4)
        query_embed3 = torch.cat([query_embed3, hs4[-1]], dim=0)

        tgt3 = torch.zeros_like(query_embed3)
        hs3 = self.decoder3(tgt3, ctx3, tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_mask3, pos=pos_embed3, query_pos=query_embed3)
        query_embed2 = torch.cat([query_embed2, hs3[-1]], dim=0)

        tgt2 = torch.zeros_like(query_embed2)
        hs2 = self.decoder2(tgt2, ctx2, tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_mask2, pos=pos_embed2, query_pos=query_embed2)
        query_embed1 = torch.cat([query_embed1, hs2[-1]], dim=0)

        tgt1 = torch.zeros_like(query_embed1)
        hs = self.decoder1(tgt1, ctx1, tgt_mask=tgt_mask,
                           memory_key_padding_mask=src_mask1, pos=pos_embed1, query_pos=query_embed1)

        return hs.transpose(1, 2)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0), edge


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")