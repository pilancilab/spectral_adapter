import math

import torch
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F
import numpy as np

if is_xformers_available():
    import xformers
    

def remove_edlora_unet_attention_forward(unet):
    def change_forward(unet):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and name == 'attn2':
                layer.set_processor(AttnProcessor())
            else:
                change_forward(layer)
    change_forward(unet)


class EDLoRA_Control_AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, cross_attention_idx, place_in_unet, controller, attention_op=None):
        self.cross_attention_idx = cross_attention_idx
        self.place_in_unet = place_in_unet
        self.controller = controller
        self.attention_op = attention_op

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available() and not is_cross:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class EDLoRA_AttnProcessor:
    def __init__(self, cross_attention_idx, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def revise_edlora_unet_attention_forward(unet):
    def change_forward(unet, count):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                layer.set_processor(EDLoRA_AttnProcessor(count))
                count += 1
            else:
                count = change_forward(layer, count)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx)
    print(f'Number of attention layer registered {cross_attention_idx}')


def revise_edlora_unet_attention_controller_forward(unet, controller):
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def change_forward(unet, count, place_in_unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:  # only register controller for cross-attention
                layer.set_processor(EDLoRA_Control_AttnProcessor(count, place_in_unet, controller))
                count += 1
            else:
                count = change_forward(layer, count, place_in_unet)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0, 'down')
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx, 'mid')
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx, 'up')
    print(f'Number of attention layer registered {cross_attention_idx}')
    controller.num_att_layers = cross_attention_idx

class SpectralLinearLayer_OFT(nn.Module):
    def __init__(self, name, original_module, rank=4, alpha=1, top=True, idx=0, revised_r=-1):
        rank = 8
        super().__init__()
        self.name = name
        if original_module.__class__.__name__ == 'Conv2d':
            self.conv = True
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
        else:
            self.conv = False
            in_channels, out_channels = original_module.in_features, original_module.out_features
        W = original_module.weight.data.view(out_channels, in_channels)
        U, S, V = torch.svd(W)
        self.U = torch.nn.Parameter(U, requires_grad=False)
        self.S = torch.nn.Parameter(S, requires_grad=False)
        self.V = torch.nn.Parameter(V, requires_grad=False)
        self.spectral_A = torch.nn.Parameter(torch.zeros(revised_r,revised_r), requires_grad=True)
        self.spectral_B = torch.nn.Parameter(torch.zeros(revised_r,revised_r), requires_grad=True)
        self.spectral_C = torch.nn.Parameter(torch.ones(revised_r), requires_grad=True)
        original_module.forward = self.forward
        self.original_module = original_module
        self.top = top
        self.idx = idx
        assert revised_r>0
        self.rank = revised_r

    def cayley(self, data: torch.Tensor) -> torch.Tensor:
        r, _ = data.shape
        skew = 0.5 * (data - data.T)
        I = torch.eye(r, device=data.device)
        Q = torch.mm(I - skew, torch.inverse(I + skew))
        return Q

    def forward(self, hidden_states):
        if self.top:
            pad_U = self.U.clone()
            pad_U[:,self.idx*self.rank:(self.idx+1)*self.rank] = self.U[:,self.idx*self.rank:(self.idx+1)*self.rank]@self.cayley(self.spectral_A)
            pad_S = self.S.clone()
            pad_S[self.idx*self.rank:(self.idx+1)*self.rank] = self.S[self.idx*self.rank:(self.idx+1)*self.rank]*self.spectral_C
            pad_V = self.V.clone()
            pad_V[:,self.idx*self.rank:(self.idx+1)*self.rank] = self.V[:,self.idx*self.rank:(self.idx+1)*self.rank]@self.cayley(self.spectral_B)
        else:
            raise Exception('')
        pad_W = pad_U@pad_S.diag()@pad_V.T
        if self.conv :
            raise Exception('')
        else:
            return F.linear(hidden_states, pad_W, bias=self.original_module.bias)

