import logging
import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from torchvision.ops import roi_align

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange

import sys

sys.path.append("/media/sdc/fe/flash-attention/flash_attn")

from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from .flash_attention_class import FlashAttention
from flash_attn.modules.mlp import FusedMLP
from flash_attn.ops.rms_norm import DropoutAddRMSNorm
from flash_attn.modules.mha import MHA, FlashCrossAttention

logger = logging.getLogger(__name__)

class AdapterConv3D(nn.Module):
    """
    视频版的 3D 卷积 Adapter。
    假设在某个 stage 内的特征 (B, N, C)，其中 N = T * H * W。
    需要在外部知道当前 T/H/W 来做 reshape。
    """
    def __init__(self, in_channels, adapter_channels, T, H, W, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.conv = nn.Conv3d(
            in_channels, 
            adapter_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm3d(adapter_channels, momentum=0.01)
        self.act = nn.ReLU(inplace=True)
        # 投影回原通道
        self.conv_proj = nn.Conv3d(adapter_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, N, C) 其中 N=T*H*W
        return: (B, N, C) 形状一致，但已经融合了局部3D卷积的信息
        """
        B, N, C = x.shape
        # (B, N, C) -> (B, C, T, H, W)
        x_3d = x.transpose(1, 2).reshape(B, C, self.T, self.H, self.W)
        
        # 3D 卷积
        out_3d = self.conv(x_3d)     # (B, adapter_channels, T, H, W)
        out_3d = self.bn(out_3d)
        out_3d = self.act(out_3d)

        # 投影回原通道
        out_3d = self.conv_proj(out_3d)  # (B, C, T, H, W)

        # reshape 回 (B, N, C)
        out = x + out_3d.reshape(B, C, -1).transpose(1, 2)  # (B, N, C)
        return out
    
class BlockWithAdapter(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., T=8, H=16, W=16, adapter_ratio=0.25,
                 use_flash_attn=False, use_fused_mlp=False, with_cp=False, qk_normalizationg=False,
                   layerscale_no_force_fp32=False, use_fused_rmsnorm=False,  **kwargs):
        super().__init__()
        self.block = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, **kwargs)
        adapter_channels = int(dim * adapter_ratio)
        self.adapter_3d = AdapterConv3D(
            in_channels=dim,
            adapter_channels=adapter_channels,
            T=T, H=H, W=W
        )
    @property
    def mlp(self):
        return self.block.mlp
    
    @property
    def norm1(self):
        return self.block.norm1
    
    @property
    def attn(self):
        return self.block.attn
    
    @property
    def ls1(self):
        return self.block.ls1
    
    @property
    def drop_path1(self):
        return self.block.drop_path1
    
    @property
    def norm2(self):
        return self.block.norm2
    
    @property
    def ls2(self):
        return self.block.ls2
    
    @property
    def drop_path2(self):
        return self.block.drop_path2
    
    @property
    def with_cp(self):
        return self.block.with_cp
    
    def forward(self, x, residual):
        x = self.block(x, residual)

        # 脱离cls_token并保存
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        patch_tokens = patch_tokens + self.adapter_3d(patch_tokens)
        x = torch.cat([cls_token, patch_tokens], dim=1) # (B, T * H * W, C)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1,
            proj_drop=0.1, attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_proj = nn.LayerNorm(out_dim)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.norm_proj(x)
        
        return x


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        if drop_path > 0.:
            logger.info(f"Use DropPath in projector: {drop_path}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        
        return x


class AttentionPoolingBlock(AttentiveBlock):
    
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)
        
        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)
        
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs
    
    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def forward(self, x, residual=None):
        
        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x
        
        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)

class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
            num_frames=8, tubelet_size=1, norm_layer=None, use_flash_attn=False
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        ) # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.use_flash_attn = True
        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]), 
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def Roi_Align_M(self, x):

        B, C, T, H, W = x.shape
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        x = self.norm(x)
        return x

class PatchEmbed3D(nn.Module):
    """
    与原先的 PatchEmbed 类似，但把它做成可配置化的单分支 3D 卷积 patch embed。
    不与原先的特征融合，单独返回
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768, 
                 num_frames=8, tubelet_size=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        )  # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Conv3d for patch embedding
        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )

        # self.norm = nn.LayerNorm(embed_dim)

        self.roi_attn = CrossAttention(dim=1024, num_heads=8, qkv_bias=True)
        self.global_attn = CrossAttention(dim=1024, num_heads=8, qkv_bias=True)

    def Roi_Align_M(self, x, roi_box_list, Hr, Wr):

        B, C, T, H, W = x.shape
        _, _, num_box, box_size = roi_box_list.shape
        x_to_align = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        box_to_align = roi_box_list.contiguous().view(B*T*num_box, box_size)

        boxes_list = []
        for b in range(B*T):
            for n in range(num_box):
                idx = torch.tensor([b], dtype=x.dtype).to(x.device)
                box = torch.cat((idx, box_to_align[b*num_box + n]), dim=0)
                boxes_list.append(box)
            
        boxes_list = torch.stack(boxes_list)
        output_size = (Hr, Wr)
        roi_features = roi_align(
            x_to_align,
            boxes_list,
            output_size,
            spatial_scale= 16.0 / 224,
            sampling_ratio=0,
            aligned=True
        )

        roi_features = roi_features.view(B*T, num_box, C, Hr, Wr).mean(dim=1)
        roi_features = roi_features.view(B, T, C, Hr, Wr).permute(0, 2, 1, 3, 4).contiguous()
        return  roi_features
    
    def forward(self, x, roi_box_list):
        """
        x: (B, C, T, H, W)
        return: (B, T*H*W, C), plus shape info
        """
        Hr, Wr = 16, 16
        x = self.proj(x)

        B, C, T, H, W = x.shape
        
        mask_label = torch.tensor([[  0.,  0.,  224., 224.]], device=f"{x.device}", dtype=torch.bfloat16)
        mask_list = [0] * (B * T)
        for b, bs in enumerate(roi_box_list):
            for f, frame in enumerate(bs):
                if torch.equal(frame, mask_label):
                    mask_list[b*T+f] = 1
    
        roi_feature = self.Roi_Align_M(x, roi_box_list, Hr, Wr)
        roi_input = roi_feature.permute(0, 2, 3, 4, 1).contiguous().view(B*T, Hr*Wr, C)
        x_input = x.permute(0, 2, 3, 4, 1).contiguous().view(B*T, H*W, C)

        attn_feature = self.roi_attn(roi_input, x_input, x_input)
        global_attn_input = self.global_attn(x_input, attn_feature, attn_feature)

        global_attn_input = global_attn_input.view(B, T, H*W, C)
        return global_attn_input, (T, H, W)

class MultiScalePatchEmbed(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 in_chans=3,
                 embed_dim_high=1024,
                 embed_dim_low=768,
                 patch_size_high=14,
                 patch_size_low=16,
                 num_frames=8,
                 tubelet_size=1,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed_high = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size_high,
            in_chans=in_chans,
            embed_dim=embed_dim_high,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            norm_layer=norm_layer
        )

        self.img_size = img_size
        self.patch_size_high = patch_size_high

    def forward(self, x, roi_box_list):
        x_high, shape_high = self.patch_embed_high(x, roi_box_list)
        return x_high, shape_high

class CrossScaleFusion(nn.Module):
    """
    同时让 high_res token 与 low_res token 进行交互的模块。
    - 先 high->low cross-attn
    - 再 low->high cross-attn
    - 最终输出融合后的 x_high, x_low
    """
    def __init__(
        self, 
        dim_high=1024,
        dim_low=768,
        num_heads_high=8,
        num_heads_low=8,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        
        # high->low cross-attn: Q=high, K/V=low
        self.high2low_attn = CrossAttention(dim=dim_high, num_heads=num_heads_high, 
                                            attn_drop=attn_drop, proj_drop=proj_drop)
        
        # low->high cross-attn: Q=low, K/V=high
        self.low2high_attn = CrossAttention(dim=dim_low, num_heads=num_heads_low, 
                                            attn_drop=attn_drop, proj_drop=proj_drop)

        # 如果需要残差 + Norm，可以再加 LayerNorm、DropPath、MLP等
        self.norm_high = nn.LayerNorm(dim_high)
        self.norm_low  = nn.LayerNorm(dim_low)
        # 也可以添加 MLP adapter 等，这里省略

    def forward(self, x_high, x_low):
        """
        x_high: (B, N_high, dim_high)
        x_low:  (B, N_low,  dim_low)
        """
        # 让 high 看 low
        h2l = self.high2low_attn(self.norm_high(x_high), 
                                 self.norm_low(x_low),
                                 self.norm_low(x_low))  # (B, N_high, dim_high)
        x_high_fused = x_high + h2l 
        # 让 low 看 high
        l2h = self.low2high_attn(self.norm_low(x_low), 
                                 self.norm_high(x_high),
                                 self.norm_high(x_high))  # (B, N_low, dim_low)
        x_low_fused = x_low + l2h
        
        return x_high_fused, x_low_fused

class CrossScaleInject(nn.Module):
    """
    只做单向 Cross-Attention: 让 x_low 看到 x_high
    x_low_fused = CrossAttn(Q=x_low, K=x_high, V=x_high)
    """
    def __init__(self, dim_low, dim_high, num_heads_low=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim_low = dim_low
        self.dim_high = dim_high
        self.cross_attn = CrossAttention(dim=dim_low, num_heads=num_heads_low,
                                         attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm_low = nn.LayerNorm(dim_low)
        self.norm_high = nn.LayerNorm(dim_high)
        # 如果 dim_low != dim_high，需要先把 x_high 投影到 dim_low
        if dim_high != dim_low:
            self.adapter_proj = nn.Linear(dim_high, dim_low, bias=False)
        else:
            self.adapter_proj = nn.Identity()

    def forward(self, x_low, x_high):

        # 先norm
        q = self.norm_low(x_low)
        kv = self.norm_high(x_high)

        # 如果 high/low 的通道不一致, 先把 high 投影到与 low 相同的 dim
        kv = self.adapter_proj(kv)

        # cross-attn
        x_low_fused = self.cross_attn(q, kv, kv)  # (B, N_low, dim_low)
        # 残差
        x_low_fused = x_low_fused
        return x_low_fused
    
class getROI(nn.Module):
    def __init__(self,
        in_chans=3,
        patch_size=14,
        # patch_size = 7,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.,
        head_drop_path_rate=0.,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4,
        init_values=0.1,
        qk_normalization=True,
        depth=24,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        drop_cls_token=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
        align_dim=512,
                ):
        super().__init__()
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, logger.info(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')
        logger.info(mlp_ratio)
        
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.T = num_frames // tubelet_size
        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        logger.info(f"Droppath rate: {dpr}")
        logger.info(f"Checkpoint list: {with_cp_list}")

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])
        
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=head_drop_path_rate, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim
        )

        self.fc_norm = nn.LayerNorm(768)

        # self.fusion_inject_module = CrossScaleInject(dim_low=1024, dim_high=1024)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype
    
    def forward(self, x, roi_box_list, pos_embed, use_image=False):
        x, _= self.patch_embed(x.type(self.dtype), roi_box_list)
        B, T, Size, C = x.shape
        x = x.view(B, T*Size, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed

        residual = None
        for blk in self.blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        
        x = self.clip_projector(x)

        x = self.fc_norm(x)
        return x
        

class InternVideo2_roi_branch(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25, # may need ablation
            head_drop_path_rate: float = 0.,
            embed_dim: int = 1408,
            num_heads: int = 16,
            mlp_ratio: float = 48/11,
            init_values: float = 1e-5, # may need ablation
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False, # when True for training?
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            embed_dim_high: int = 1024,
            checkpoint_num: int = 0,
            patch_size_high: int = 14,
            num_heads_high=8,
            num_heads_low=8,
            attn_drop=0.,
            proj_drop=0.,
        ):
        super().__init__()

        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, logger.info(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')
        logger.info(mlp_ratio)
        
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.T = num_frames // tubelet_size
        
        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            logger.info("Use seperable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            logger.info("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        logger.info(f"Droppath rate: {dpr}")
        logger.info(f"Checkpoint list: {with_cp_list}")
        
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])

        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=head_drop_path_rate, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim
        )

        self.fc_norm = nn.Identity()

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.roi_branch = getROI()

        # self.fusion_inject_module = CrossScaleInject(dim_low=1024, dim_high=1024)

    def init_pos_embed(self):
        logger.info("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            # trunc_normal_(self.pos_embed_spatial, std=.02)
            # trunc_normal_(self.pos_embed_temporal, std=.02)
            # trunc_normal_(self.pos_embed_cls, std=.02)
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
            )
            self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[0], # t_size
            )
            self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
        else:
            # trunc_normal_(self.pos_embed, std=.02)
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
                self.patch_embed.grid_size[0], # t_size
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 
            'pos_embed_spatial', 
            'pos_embed_temporal', 
            'pos_embed_cls',
            'cls_token'
        }
    
    def forward(self, x, roi_box_list, use_image=False):
        ori_x = x
        x = self.patch_embed(x.type(self.dtype))

        B, T, L, C = x.shape  # T: temporal; L: spatial
        x = x.view([B, T * L, C])

        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add pos_embed
        if self.sep_pos_embed:
            if use_image:
                pos_embed = self.pos_embed_spatial
            else:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.grid_size[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.grid_size[1] * self.grid_size[2],
                    dim=1,
                )
            pos_embed = torch.cat(
                [
                    self.pos_embed_cls.expand(pos_embed.shape[0], -1, -1),
                    pos_embed,
                ],
                1,
            )
        else:
            if use_image:
                cls_pos_embed = self.pos_embed[:, :1, :]
                img_pos_embed = self.pos_embed[:, 1:, :].view(1, self.T, L, C).mean(dim=1)
                pos_embed = torch.cat([cls_pos_embed, img_pos_embed], dim=1)
            else:
                pos_embed = self.pos_embed

        roi_feature = self.roi_branch(ori_x, roi_box_list, pos_embed)
        x = x + pos_embed

        residual = None
        for blk in self.blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        
        x = self.clip_projector(x)

        x = self.fc_norm(x)
        return x, roi_feature
