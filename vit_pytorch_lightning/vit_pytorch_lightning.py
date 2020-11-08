import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat

MIX_NUM_PATCHES = 16

# Residual Module
class Residual(nn.Module):
    def __init__(self, fn):
      super().__init__()
      self.fn = fn

    def forward(self, x, **kwags):
      return self.fn(x, **kwags) + x

# Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwags)

# MLP
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
          nn.Linear(dim, hidden_dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(hidden_dim, dim),
          nn.Dropout(dropout)
        )
    def forward(seld, x):
        return self.net(x)

# Multi-Head Attention
class Attention(nn.Module):
  def __init__(self, dim, heads=8, dropout=0.):
      super().__init__()
      self.heads = heads
      self.scale = dim ** -0.5

      self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
      self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout)
      )

  def forward(self, x, mask=None):
      b, n, _, h = *x.shape, self.heads
      qkv = self.to_qkv(x).chunk(3, dim=-1)
      q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

      dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

      if mask is not None:
          mask = F.pad(mask.flatten(1), (1, 0), value=True)
          assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
          mask = mask[:, None, :] * mask[:, :, None]
          dots.masked_fill_(~mask, float('-inf'))
          del mask

      attn = dots.softmax(dim=-1)

      out = torch.einsum('bhij,bhjd->bhid', attn, v)
      out = rearrange(out, 'b h b d -> b n (h d)')
      out = self.to_out(out)
      return out

class Transformer(nn.Module):
  def __init__(self, dim, depth, heads, mlp_dim, dropout):
      super().__init__()
      self.layers = nn.ModuleList([])
      for _ in range(depth):
          self.layers.append(nn.ModuleList([
            Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
            Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
          ]))

  def forward(self, x, mask=None):
      for attn, ff in self.layers:
          x = attn(x, mask=mask)
          x = ff(x)
      return x

class ViT(pl.LightningModule):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0., lr=0.01):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIX_NUM_PATCHES, f'yout number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_classes+1, dim)) # num_classes+1 -> num_patch + [CLS] token
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity() # 恒等関数

        self.mlp_head = nn.Sequential(
          nn.LayerNorm(dim),
          nn.Linear(dim, mlp_dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(mlp_dim, num_classes)
        )

        self.lr = lr

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        # [cls]トークンの最終出力を受け取り、MLPのインプットとする
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y, t), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc(y, t), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', self.test_acc(y, t), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
