import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from einops import rearrange, einsum

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device

    num_tokens = mask.sum(dim = -1)
    # num_pads = seq_len - num_tokens
    num_masked = (prob * num_tokens).round().clamp(min = 1)
    # breakpoint()
    randperm = torch.rand((batch, seq_len), device = device)
    randperm[~mask] = 100
    randperm_indices = randperm.argsort(dim = -1)
    # randperm_indices -= rearrange(num_pads, 'b -> b 1')

    # breakpoint()
    # randperm_indices.masked_fill_(randperm_indices < 0, seq_len) # set to max out of bounds, so never chosen
    # breakpoint()
    mask_subset = randperm_indices < rearrange(num_masked, 'b -> b 1')
    return mask_subset

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                has_cross_attn=True):
        super().__init__()

        self.transformer = CrossCondTransformer(num_vq, embed_dim, clip_dim, block_size, num_layers*2, n_head, drop_out_rate, fc_rate, has_cross_attn=has_cross_attn)
        self.block_size = block_size
        self.num_vq = num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature, token_mask, text_mask):

        # breakpoint()
        logits = self.transformer(idxs, clip_feature, token_mask=token_mask, text_mask=text_mask)
        # logits = self.trans_head(feat)
        return logits
    
    def forward_with_cond_scale(
        self,
        idxs, clip_feature,
        cond_scale = 3,
    ):
        logits = self.forward(idxs, clip_feature, token_mask=None, text_mask=None)

        if cond_scale == 1:
            return logits

        text_mask = torch.zeros_like(idxs).bool()
        null_logits = self.forward(idxs, clip_feature, token_mask=None, text_mask=text_mask)
        return null_logits + (logits - null_logits) * cond_scale

    def sample(self, clip_feature, if_categorial=False):
        for k in range(self.block_size):
            if k == 0:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs

class CrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            mask = rearrange(mask, 'b j -> b 1 1 j')
            att = att.masked_fill(~mask, -torch.finfo(att.dtype).max)
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class CrossConditionalCrossAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        # self.key = nn.Linear(embed_dim, embed_dim)
        # self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, context, mask=None):
        B, T, C = x.size() 


        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k = self.key(context).view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(context).view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = torch.ones((B, self.n_head, T, 1), dtype=torch.float, device=x.device)
        if mask is not None:
            # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            mask = rearrange(mask, 'b j -> b 1 j 1')
            # att = att.masked_fill(~mask, -torch.finfo(att.dtype).max)
            att = att.masked_fill(~mask, 0)
            
        # breakpoint()
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # breakpoint()
        # y = einsum('b h i i, b h i d -> b h i j', att, v)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4, has_cross_attn=False):
        super().__init__()

        self.has_cross_attn = has_cross_attn
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

        if self.has_cross_attn:
            self.cross_attn = CrossConditionalCrossAttention(embed_dim, block_size, n_head, drop_out_rate)
            self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None, self_attn_mask=None, cross_attn_mask=None):
        # breakpoint()
        # x = inp['x']
        # context = inp['context']

        # print(x)

        x = x + self.attn(self.ln1(x), mask=self_attn_mask)
        if self.has_cross_attn:
            x = x + self.cross_attn(self.ln3(x), context, mask=cross_attn_mask)
        x = x + self.mlp(self.ln2(x))

        return x 
        # return {'x': x, 'context': context}


class CrossCondTransformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                has_cross_attn=True):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.tok_emb = nn.Embedding(num_vq+3, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        # self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate, has_cross_attn=has_cross_attn) for _ in range(num_layers)])
        # self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq+1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, token_mask=None, text_mask=None):
        # if len(idx) == 0:
        #     token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        # else:
        #     b, t = idx.size()
        #     assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        #     # forward the Trans model
        #     token_embeddings = self.tok_emb(idx)
        #     token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
        
        # if exists(conditioning_token_ids):
        #     conditioning_token_ids = rearrange(conditioning_token_ids, 'b ... -> b (...)')
        #     cond_token_emb = self.token_emb(conditioning_token_ids)
        #     context = torch.cat((context, cond_token_emb), dim = -2)
        #     context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value = True)

        token_embeddings = self.tok_emb(idx)
        context_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        x = self.pos_embed(token_embeddings)

        # NOTE Add text condition to queries 
        if not self.has_cross_attn:
            x = x + context_embeddings

        for block in self.blocks:
            x = block(x, context=context_embeddings, self_attn_mask=token_mask, cross_attn_mask=text_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    

