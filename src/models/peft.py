# Implement lora and adapter layers
# 2023 Ruchao Fan SPAPL

# Reference code: Loralib from Microsoft
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py 

import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, model, rank, alpha, dropout):
        super(LoRALinear, self).__init__()

        self.rank = rank
        self.lora_alpha = alpha

        if not isinstance(model, nn.Linear):
            raise NotImplementedError("For Linear layer!")

        out_dim, in_dim = model.weight.size()
        self.lora_A = nn.Parameter(model.weight.new_zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(model.weight.new_zeros((out_dim, rank)))
        self.scaling = alpha / rank
        
        if dropout > 0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

class LoRAConv(nn.Module):
    def __init__(self, model, rank, alpha, dropout):
        super(LoRAConv, self).__init__()

        self.rank = rank
        self.lora_alpha = alpha

        if not isinstance(model, nn.Conv1d):
            raise NotImplementedError("For Conv1d layer!")

        in_dim = model.in_channels
        out_dim = model.out_channels
        kernel_size = model.kernel_size[0]
        groups = model.groups
        self.lora_A = nn.Parameter(model.weight.new_zeros((rank, in_dim * kernel_size)))
        self.lora_B = nn.Parameter(model.weight.new_zeros((int(out_dim/groups), rank)))
        self.scaling = alpha / rank
        
        if dropout > 0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

class ResidualAdapter(nn.Module):
    "Residual adapters."
    def __init__(self, 
        embedding_dim: float = 768, 
        bottleneck_dim: float = 1024,
        dropout: float = 0.1, 
        layer_norm_first: bool = True,
    ) -> None:

        super(ResidualAdapter, self).__init__()
        self.adapter = nn.Sequential(nn.Linear(embedding_dim, bottleneck_dim), 
                                      nn.GELU(),
                                      nn.Linear(bottleneck_dim, embedding_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.adapter_layer_norm = nn.LayerNorm(embedding_dim)

        self.size = bottleneck_dim
    
        self.reset_parameters()

    def reset_parameters(self):
        # zero initialization for the second linear layer, which is similar to the lora initialization
        for p in self.adapter[2].parameters():
            nn.init.zeros_(p)
        
    def forward(self, x: torch.Tensor):
        residual = x
    
        if self.layer_norm_first:
            x = self.adapter_layer_norm(x)
            x = residual + self.dropout(self.adapter(x))
        else:
            x = residual + self.dropout(self.adapter(x))
            x = self.adapter_layer_norm(x)

        return x

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError

def mark_only_adapter_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'adapter' not in n:
            p.requires_grad = False


        