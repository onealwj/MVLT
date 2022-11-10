# Copyright (c)  Westone Information Industry Inc.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MVLT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(112, 448), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 txt_max_len=27, num_char_class=38, iter_correct=3):
        super().__init__()

        self.img_size = img_size
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.txt_max_len = txt_max_len
        self.num_char_class = num_char_class
        self.decoder_embed_dim = decoder_embed_dim
        self.iter_correct = iter_correct

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.text_mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim))
        self.text_proj = nn.Linear(
            self.num_char_class, decoder_embed_dim, False)

        decoder_token_num = num_patches + 1
        decoder_token_num += self.txt_max_len
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, decoder_token_num, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        decoder_dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=decoder_dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_text_pred = nn.Linear(
            decoder_embed_dim, self.num_char_class, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.num_patches_h, self.num_patches_w), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        append_len = self.txt_max_len
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], (self.num_patches_h, self.num_patches_w), cls_token=True, append_len=append_len)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.text_mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'decoder_pos_embed', 'text_mask_token'}

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, text):
        batch_size = x.shape[0]
        # embed tokens
        x = self.decoder_embed(x)

        text_mask_tokens = self.text_mask_token.expand(
            batch_size, self.txt_max_len, -1)

        v = x

        t = text_mask_tokens
        x = torch.cat([v, t], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        t = x[:, -self.txt_max_len:, :]
        t = self.decoder_norm(t)
        t = self.decoder_text_pred(t)
        res = t
        res_iter_list = []
        for i in range(self.iter_correct):
            tokens = torch.softmax(t, dim=-1)
            tokens = tokens.detach()
            embed = self.text_proj(tokens)

            out = (tokens.argmax(dim=-1) == 1)

            cum = out.cumsum(-1) < 1
            cum = out.cumsum(-1) < 2
            cum[:, 0] = False
            cum = cum.unsqueeze(-1)
            cum = cum.expand(-1, -1, self.decoder_embed_dim)

            embed = torch.where(cum, embed, text_mask_tokens)

            x = torch.cat((v, embed), 1)
            x = x + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                x = blk(x)

            t = x[:, -self.txt_max_len:, :]
            t = self.decoder_norm(t)
            t = self.decoder_text_pred(t)

            res_iter_list.append(t)

        return res, res_iter_list

    def forward(self, imgs, text):
        latent = self.forward_encoder(imgs)
        res, res_iter_list = self.forward_decoder(latent, text)
        return res, res_iter_list


def mvlt_vit_base_patch16_dec512d4b(**kwargs):
    model = MVLT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mvlt_vit_large_patch16_dec512d4b(**kwargs):
    model = MVLT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mvlt_vit_huge_patch14_dec512d8b(**kwargs):
    model = MVLT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mvlt_vit_base_patch16 = mvlt_vit_base_patch16_dec512d4b  # decoder: 512 dim, 4 blocks
# decoder: 512 dim, 8 blocks
mvlt_vit_large_patch16 = mvlt_vit_large_patch16_dec512d4b
mvlt_vit_huge_patch14 = mvlt_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
