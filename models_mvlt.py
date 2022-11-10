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


class PretrainMVLT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(112, 448), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 exp_semantic=False, imp_semantic=False, txt_max_len=27, num_char_class=38):
        super().__init__()

        self.img_size = img_size
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.exp_semantic = exp_semantic
        self.imp_semantic = imp_semantic
        self.txt_max_len = txt_max_len
        self.num_char_class = num_char_class
        self.decoder_embed_dim = decoder_embed_dim

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        if self.exp_semantic or self.imp_semantic:
            self.text_mask_token = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))
            self.text_proj = nn.Linear(
                self.num_char_class, decoder_embed_dim, False)

        decoder_token_num = num_patches + 1
        if self.exp_semantic or self.imp_semantic:
            decoder_token_num += self.txt_max_len
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, decoder_token_num, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch
        self.decoder_text_pred = nn.Linear(
            decoder_embed_dim, self.num_char_class, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.num_patches_h, self.num_patches_w), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        append_len = 0
        if self.exp_semantic or self.imp_semantic:
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
        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.exp_semantic or self.imp_semantic:
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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        #h = w = int(x.shape[1]**.5)
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        #assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, text, w_mask):
        batch_size = x.shape[0]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            batch_size, ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        if self.exp_semantic or self.imp_semantic:
            v_pos_embed = self.decoder_pos_embed[:,
                                                 :self.patch_embed.num_patches+1, :]
            t_pos_embed = self.decoder_pos_embed[:,
                                                 self.patch_embed.num_patches+1:, :]
            text_mask_tokens = self.text_mask_token.expand(
                batch_size, self.txt_max_len, -1)

            x = x + v_pos_embed
            v = x

            if self.imp_semantic:
                imp_t = text_mask_tokens + t_pos_embed
                imp_x_t = torch.cat([v, imp_t], dim=1)
                x = imp_x_t

            if self.exp_semantic:
                onehot = nn.functional.one_hot(text, self.num_char_class)
                t_embed = self.text_proj(onehot.float())
                t_embed = torch.where(
                    w_mask.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), text_mask_tokens, t_embed)
                pad_mask = (text == 0)
                t_embed = torch.where(
                    pad_mask.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), text_mask_tokens, t_embed)
                exp_t = t_embed + t_pos_embed
                exp_x_t = torch.cat([v, exp_t], dim=1)
                x = exp_x_t

            if self.exp_semantic and self.imp_semantic:
                x = torch.cat([exp_x_t, imp_x_t], dim=0)

        else:
            x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        exp_x_t = None
        imp_x_t = None
        if self.exp_semantic and self.imp_semantic:
            exp_x_t = x[:batch_size, :, :]
            imp_x_t = x[batch_size:, :, :]
        elif self.exp_semantic:
            exp_x_t = x
        elif self.imp_semantic:
            imp_x_t = x

        res_dict = {"v": None, "exp_v": None,
                    "exp_t": None, "imp_v": None, "imp_t": None}
        if exp_x_t is not None or imp_x_t is not None:
            if exp_x_t is not None:
                exp_v = exp_x_t[:, :-self.txt_max_len, :]
                exp_t = exp_x_t[:, -self.txt_max_len:, :]

                exp_v = self.decoder_norm(exp_v)
                exp_t = self.decoder_norm(exp_t)

                exp_v = self.decoder_pred(exp_v)
                exp_v = exp_v[:, 1:, :]
                exp_t = self.decoder_text_pred(exp_t)

                res_dict["exp_v"] = exp_v
                res_dict["exp_t"] = exp_t
            if imp_x_t is not None:
                imp_v = imp_x_t[:, :-self.txt_max_len, :]
                imp_t = imp_x_t[:, -self.txt_max_len:, :]

                imp_v = self.decoder_norm(imp_v)
                imp_t = self.decoder_norm(imp_t)

                imp_v = self.decoder_pred(imp_v)
                imp_v = imp_v[:, 1:, :]
                imp_t = self.decoder_text_pred(imp_t)

                res_dict["imp_v"] = imp_v
                res_dict["imp_t"] = imp_t
        else:
            x = self.decoder_norm(x)

            x = self.decoder_pred(x)
            x = x[:, 1:, :]

            res_dict["v"] = x

        return res_dict

    def v_loss(self, pred, target, mask):

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def t_loss(self, pred, target):

        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(
            pred.view(-1, pred.shape[-1]), target.contiguous().view(-1))
        return loss

    def forward_loss(self, imgs, res_dict, mask, text, w_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss_dict = {"loss": None, "imp_v_loss": None,
                     "imp_t_loss": None, "exp_v_loss": None, "exp_t_loss": None}
        if res_dict["v"] is not None:
            loss = self.v_loss(res_dict["v"], target, mask)
            loss_dict["loss"] = loss
            return loss_dict

        if res_dict["imp_v"] is not None:
            imp_v_loss = self.v_loss(res_dict["imp_v"], target, mask)
            loss_dict["imp_v_loss"] = imp_v_loss
        if res_dict["imp_t"] is not None:
            imp_t_loss = self.t_loss(res_dict["imp_t"], text)
            loss_dict["imp_t_loss"] = imp_t_loss
        if res_dict["exp_v"] is not None:
            exp_v_loss = self.v_loss(res_dict["exp_v"], target, mask)
            loss_dict["exp_v_loss"] = exp_v_loss
        if res_dict["exp_t"] is not None:
            masked_text = text.clone()
            masked_text[~w_mask] = 0
            exp_t_loss = self.t_loss(res_dict["exp_t"], masked_text)
            loss_dict["exp_t_loss"] = exp_t_loss

        if loss_dict["imp_v_loss"] is not None and loss_dict["exp_v_loss"] is not None:
            v_loss = loss_dict["imp_v_loss"] * \
                0.5 + loss_dict["exp_v_loss"] * 0.5
            t_loss = loss_dict["imp_t_loss"] * \
                0.5 + loss_dict["exp_t_loss"] * 0.5
        elif loss_dict["imp_v_loss"] is not None:
            v_loss = loss_dict["imp_v_loss"]
            t_loss = loss_dict["imp_t_loss"]
        elif loss_dict["exp_v_loss"] is not None:
            v_loss = loss_dict["exp_v_loss"]
            t_loss = loss_dict["exp_t_loss"]

        loss = v_loss + 0.002*t_loss
        loss_dict["loss"] = loss
        return loss_dict

    def forward(self, imgs, text, w_mask, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        res_dict = self.forward_decoder(latent, ids_restore, text, w_mask)
        loss_dict = self.forward_loss(imgs, res_dict, mask, text, w_mask)
        return loss_dict, res_dict, mask


def mvlt_vit_base_patch16_dec512d4b(**kwargs):
    model = PretrainMVLT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mvlt_vit_large_patch16_dec512d4b(**kwargs):
    model = PretrainMVLT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mvlt_vit_huge_patch14_dec512d8b(**kwargs):
    model = PretrainMVLT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mvlt_vit_base_patch16 = mvlt_vit_base_patch16_dec512d4b  # decoder: 512 dim, 4 blocks
# decoder: 512 dim, 8 blocks
mvlt_vit_large_patch16 = mvlt_vit_large_patch16_dec512d4b
mvlt_vit_huge_patch14 = mvlt_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
