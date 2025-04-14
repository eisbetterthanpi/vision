# @title ViT
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_module(module):
    for p in module.parameters(): p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3):
        super().__init__()
        out_ch = out_ch or in_ch
        act = nn.SiLU() #
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.block = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act,
        #     zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)), nn.BatchNorm2d(out_ch), act,
            nn.BatchNorm2d(in_ch), act, nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm2d(out_ch), act, zero_module(nn.Conv2d(out_ch, out_ch, kernel, padding=kernel//2)),
            )

    def forward(self, x): # [b,c,h,w]
        return self.block(x) + self.res_conv(x)


class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=1, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        d_model = 64
        if self.r>1: self.net = nn.Sequential(ResBlock(in_ch, out_ch*r**2, kernel), nn.PixelShuffle(r))
        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), ResBlock(in_ch*r**2, out_ch, kernel))
        elif in_ch != out_ch: self.net = ResBlock(in_ch, out_ch, kernel)
        else: self.net = lambda x: torch.zeros_like(x)

    def forward(self, x):
        return self.net(x)


def adaptive_avg_pool_nd(n, x, output_size): return [nn.Identity, F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d][n](x, output_size)
def adaptive_max_pool_nd(n, x, output_size): return [nn.Identity, F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d][n](x, output_size)

def adaptive_pool_at(x, dim, output_size, pool='avg'): # [b,c,h,w]
    x = x.transpose(dim,-1)
    shape = x.shape
    parent={'avg':F.adaptive_avg_pool1d, 'max':F.adaptive_max_pool1d}[pool]
    return parent(x.flatten(0,-2), output_size).unflatten(0, shape[:-1]).transpose(dim,-1)


class ZeroExtend():
    def __init__(self, dim=1, output_size=16):
        self.dim, self.out = dim, output_size
    def __call__(self, x): # [b,c,h,w]
        return torch.cat((x, torch.zeros(*x.shape[:self.dim], self.out - x.shape[self.dim], *x.shape[self.dim+1:], device=device)), dim=self.dim)

def shortcut_fn(x, dim=1, c=3, sp=(3,3), nd=2):
    x = adaptive_pool_at(x, dim, c, pool='max')
    x = adaptive_max_pool_nd(nd, x, sp)
    return x


class UpDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, r=1):
        super().__init__()
        act = nn.SiLU()
        self.r = r
        self.block = PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)

    def forward(self, x): # [b,c,h,w]
        # b, num_tok, c, *win = x.shape
        # x = x.flatten(0,1)
        out = self.block(x)
        shortcut = shortcut_fn(x, dim=1, c=out.shape[1], sp=out.shape[-2:], nd=2)
        out = out + shortcut
        # out = out.unflatten(0, (b, num_tok))
        return out


class SelfAttn(nn.Module):
    def __init__(self, in_dim, n_heads, d_head=None):
        super().__init__()
        self.n_heads = n_heads
        d_head = d_head or in_dim//n_heads
        d_model = n_heads*d_head
        self.qkv = nn.Linear(in_dim, d_model*3, bias=False)
        self.lin = zero_module(nn.Linear(d_model, in_dim))
        self.scale = d_head**-.5

    def forward(self, x): # [b,t,d]
        q,k,v = self.qkv(x).unflatten(-1, (self.n_heads,-1)).transpose(-3,-2).chunk(3, dim=-1) # [b, r^2, h/r*w/r, dim] # [b*r^2, h/r*w/r, n_heads, d_head]?
        q, k = q.softmax(dim=-1)*self.scale, k.softmax(dim=-2)
        context = k.transpose(-2,-1) @ v # [batch, n_heads, d_head, d_head]
        x = q @ context # [batch, n_heads, T/num_tok, d_head]
        x = x.transpose(1,2).flatten(2)
        return self.lin(x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, drop=0):
        super().__init__()
        self.d_model = d_model
        self.norm1, self.norm2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model) # LayerNorm RMSNorm
        self.drop = nn.Dropout(drop)
        self.attn = SelfAttn(d_model, n_heads) # 16448
        # self.attn = SelfAttn(d_model, n_heads, 8) # 16448
        # self.attn = GLAblock(hidden_size=d_model, expand_k=1, expand_v=1, num_heads=n_heads)
        # self.self = Pooling()
        act = nn.GELU() # ReLU GELU
        # self.ff = nn.Sequential(
        #     *[nn.BatchNorm2d(d_model), act, SeparableConv2d(d_model, d_model),]*3
        #     )
        # self.ff = ResBlock(d_model, kernel=1) # 74112
        # self.ff = UIB(d_model, mult=4) # uib m4 36992, m2 18944
        # self.ff = GLU(d_model, int(3.5*d_model)) # 3.5
        # self.ff = GLU(d_model, d_model) # 3.5
        ff_dim=d_model*4#mult
        self.ff = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, ff_dim), act,
            nn.RMSNorm(ff_dim), nn.Dropout(drop), zero_module(nn.Linear(ff_dim, d_model))
            # nn.RMSNorm(d_model), act, nn.Linear(d_model, ff_dim),
            # nn.RMSNorm(ff_dim), act, nn.Linear(ff_dim, d_model)
        )

    def forward(self, x, mask=None): # [b,c,h,w], [batch, num_tok, cond_dim], [batch,T]
        bchw = x.shape
        x = x.flatten(2).transpose(1,2) # [b,h*w,c]
        # print('attnblk fwd',x.shape)

        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.ff(x)
        x = x.transpose(1,2).reshape(*bchw)
        # x = self.ff(x)
        # x = x + self.drop(self.ff(self.norm2(x)))
        return x


class SimpleViT(nn.Module):
    def __init__(self, in_dim, d_model, out_dim=None, n_heads=4, nlyrs=1):
        super().__init__()
        self.embed = nn.Sequential( # in, out, kernel, stride, pad
            # nn.Conv2d(in_dim, d_model, 7, 2, 7//2, bias=False), nn.MaxPool2d(3,2,1), # nn.MaxPool2d(2,2)
            # nn.Conv2d(in_dim, d_model,3,2,3//2), nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(2,2),
            # nn.Conv2d(d_model, d_model,3,2,3//2),
            UpDownBlock(in_dim, d_model//2, r=1/2, kernel=7), UpDownBlock(d_model//2, d_model, r=1/2, kernel=3)
            # nn.PixelUnshuffle(2), nn.Conv2d(in_dim*2**2, d_model, 1, bias=False),
            )
        # self.pos_emb = LearnedRoPE2D(dim) # LearnedRoPE2D, RoPE2D
        self.pos_emb = nn.Parameter(torch.zeros(1, d_model, 8,8)) # positional_embedding == 'learnable'
        self.transformer = nn.Sequential(*[AttentionBlock(d_model, n_heads) for _ in range(nlyrs)])
        self.attn_pool = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, out_dim or d_model, bias=False)

    def forward(self, img, mask=None):
        # device = img.device
        x = self.embed(img)
        # x = self.pos_emb(x)
        bchw = x.shape
        x = x + self.pos_emb
        # for blk in self.transformer: x = blk(x)
        x = self.transformer(x)
        x = x.flatten(2).transpose(1,2) # [b,h*w,c]
        attn = self.attn_pool(x).squeeze(-1) # [batch, (h,w)] # seq_pool
        x = (attn.softmax(dim=1).unsqueeze(1) @ x).squeeze(1) # [batch, 1, (h,w)] @ [batch, (h,w), dim] -> [batch, dim]
        return self.out(x)


dim = 64
in_dim=3
out_dim = 10
model = SimpleViT(in_dim, 64, out_dim, nlyrs=1, n_heads=8).to(device) # 64129
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 59850
print(sum(p.numel() for p in model.transformer[0].attn.parameters() if p.requires_grad)) # 59850
print(sum(p.numel() for p in model.transformer[0].ff.parameters() if p.requires_grad)) # 59850
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

# print(images.shape) # [batch, 3, 32, 32]
x = torch.rand(24, 3, 32,32, device=device)
# x = torch.rand(64, 3, 28,28, device=device)
logits = model(x)
print(logits.shape)

