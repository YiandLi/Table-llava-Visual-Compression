import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x
    
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


import torch
import torch.nn as nn
from einops import rearrange


class AvgPoolProjector(nn.Module):  # mlp2x_gelu # model.get_model().mm_projector
    def __init__(
            self,
            config,
            query_num: int = 144  # for openai/clip-vit-large-patch14-336
    ):
        super().__init__()
        self.config = config
        self.query_num = query_num
        self.hw = int(self.query_num ** 0.5)  # 取一半
        self.mm_hidden_size = self.config.mm_hidden_size
        self.llm_hidden_size = self.config.hidden_size
        self.build_net()
    
    def build_net(self):
        hw = int(self.query_num ** 0.5)  # 取一半
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        # sampler = nn.AvgPool2d((self.hw, self.hw))
        self.sampler = sampler
        
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu_deco$', self.config.mm_projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
        self.mlp_projector = nn.Sequential(*modules)
    
    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        # batch_size, seq_len, h_dim = visual_feat.shape  # 576
        # hw = int(seq_len ** 0.5)  # 24
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=24,
                                       w=24)  # torch.Size([64, 1024, 24, 24])
        pooled_visual_feat = self.sampler(shaped_visual_feat)  # torch.Size([64, 1024, 12, 12])
        reshaped_visual_feat = rearrange(pooled_visual_feat, "b d h w -> b (h w) d")  # [64, 144, 1024]
        output_feat = self.mlp_projector(reshaped_visual_feat)  # [64, 144, 4096])
        return output_feat


class SVDProjector(nn.Module):
    def __init__(self,
                 mm_hidden_size: int = 1024,
                 llm_hidden_size: int = 4096,
                 reduce_n=24,  # 576 ** 0.5
                 visual_token_num=576,
                 ):
        self.reduce_n = reduce_n
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.mlp_projector = nn.Linear(self.mm_hidden_size, self.llm_hidden_size)
    
    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, h_dim = visual_feat.shape  # 576
        visual_feat = visual_feat.transpose(-1, -2).contiguous()  # batch_size, hidden_state, visual_token_num
        
        U_batch = torch.stack([
            torch.linalg.svd(visual_feat[i], full_matrices=False)[0][:, :32]
            for i in range(batch_size)])
        
        output = U_batch.transpose(-1, -2).contiguous()
        output = self.mlp_projector(output)
        return output


class Pooling2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampler = nn.AdaptiveAvgPool2d((12, 12))
    
    def forward(self, visual_feat):
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=24,
                                       w=24)  # torch.Size([64, 1024, 24, 24])
        pooled_visual_feat = self.sampler(shaped_visual_feat)  # torch.Size([64, 1024, 12, 12])
        reshaped_visual_feat = rearrange(pooled_visual_feat, "b d h w -> b (h w) d")  # [64, 144, 1024]
        return reshaped_visual_feat


class PoolingSVD(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep_patch_num = 345
    
    def forward(self, visual_feat):
        batch_size, seq_len, h_dim = visual_feat.shape  # 576
        visual_feat = visual_feat.transpose(-1, -2).contiguous()  # batch_size, hidden_state, visual_token_num
        
        visual_feat = visual_feat.float()
        U_batch = torch.stack([
            torch.linalg.svd(visual_feat[i], full_matrices=False)[0][:, :self.keep_patch_num]
            for i in range(batch_size)])
        U_batch = U_batch.half()
        
        output = U_batch.transpose(-1, -2).contiguous()
        return output


class PoolingMonkey(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep_patch_num = 345
    
    def self_soft_matching(
            self,
            metric: torch.Tensor):
        t = metric.shape[1]
        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., :, :], metric[..., :, :]
            scores = a @ b.transpose(-1, -2)  # batch_size, token_num, token_num
            b, _, _ = scores.shape
            scores_diag = torch.tril(torch.ones(t, t)) * 2
            scores_diag = scores_diag.expand(b, -1, -1).to(metric.device)
            
            scores = scores - scores_diag  # scores_diag 起到了 mask 的作用，每一个 token 只去和后面的 token 比较相似度。
            node_max, node_idx = scores.max(dim=-1)  # 每一个 token 和其 后面的token 的最大相似度
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # a中相似度排序并得到idx，降序
            
            unm_idx = edge_idx[..., t - self.keep_patch_num:, :]  # 保留的 token 的 id # batch_size, kept_num, 1
        
        def merge(src: torch.Tensor) -> torch.Tensor:
            n, t1, c = src.shape
            unm = src.gather(dim=-2,
                             index=unm_idx.expand(n, self.keep_patch_num, c))  # batch_size, kept_num, hidden_dim
            
            # 下面是还原原本的顺序，比如 kept_id 为 [12,1,9] , 这里变为 [1,9,12]
            unm_idx_new = unm_idx  # batch_size, kept_num, 1
            all_idx = unm_idx_new
            all_max, all_idx_idx = torch.sort(all_idx, dim=1)
            return unm.gather(dim=-2, index=all_idx_idx.expand(n, self.keep_patch_num, c))
        
        return merge
    
    def forward(self, visual_feat):
        merge = self.self_soft_matching(visual_feat)  # Replace with your features and r value
        latents = merge(visual_feat)
        return latents


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    # elif projector_type == 'mlp2x_gelu_deco':
    #     mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu_deco$', projector_type)
    #     mlp_depth = int(mlp_gelu_match.group(1))
    #     modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #     return nn.Sequential(*modules)
    #     # return AvgPoolProjector(config)
    
    # 重写
    # mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    mlp_gelu_match = 2
    if mlp_gelu_match:
        # mlp_depth = int(mlp_gelu_match.group(1))
        mlp_depth = 2
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type == 'identity':
        return IdentityMap()
    
    raise ValueError(f'Unknown projector type: {projector_type}')
