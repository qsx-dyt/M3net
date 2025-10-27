import torch
from einops import repeat
from torch import nn
import torch.nn.functional as F

from models.M3net import M3net


class M3net_for_SimMIM(M3net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.spatial_mask_token = nn.Parameter(torch.zeros(1, self.patch_dim, 1))

    def forward(self, x, spatial_mask=None):
        B, C, L = x.shape
        # 空间掩码
        spatial_mask_tokens = self.spatial_mask_token.expand(B, -1, L)
        w = spatial_mask.unsqueeze(1).type_as(spatial_mask_tokens) # b, 1, l
        x = x * (1. - w) + spatial_mask_tokens * w

        x = self.MSTE(x)
        # x = x.transpose(1, 2)  # [b,c,l] -> [b,l,c]
        # x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape
        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.CAT(x)
        x = self.norm(x)

        x = x[:, 1:]
        x = x.transpose(1, 2)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Conv1d(encoder.embed_dim, encoder.patch_dim, kernel_size=1),
        )

    def forward(self, x, spatial_mask):
        z = self.encoder(x, spatial_mask) #z:[b,dim,n]
        x_rec = self.decoder(z)
        loss_recon = F.l1_loss(x, x_rec, reduction='none')

        # 仅计算掩码区域的损失
        mask = torch.zeros_like(loss_recon)
        spatial_mask = spatial_mask.unsqueeze(1).float()  # [B,1,L]
        mask += spatial_mask

        loss_mask = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
        return loss_mask


def build_simmim(args, **kwargs):
    encoder = M3net_for_SimMIM(
        patch_size=args.patch_size,
        patch_dim=kwargs['band'],
        num_classes=kwargs['num_classes'],
        embed_dim=96,
        depth=kwargs['depth'],
        heads=kwargs['heads'],
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
    )
    model = SimMIM(encoder=encoder)
    return model
