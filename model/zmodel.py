import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, List, Tuple

import math
from transformers import PretrainedConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.activations import ACT2FN

class ZModelConfig(PretrainedConfig):
    model_type = "zmodel"
    def __init__(self,
                 dropout: float = 0.0,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 hidden_act: str = 'silu',
                 hidden_size: int = 512,
                 intermediate_size: int = None,
                 max_position_embeddings: int = 32768,
                 num_attention_heads: int = 8,
                 num_hidden_layers: int = 8,
                 num_kv_heads: int = 2,
                 vocab_size: int = 6400,
                 rms_norm_eps: float = 1e-5,
                 rope_theta: float = 1000000.0,
                 inference_rope_scaling: bool = False,
                 flash_attn: bool = True,
                 ################ MoE settings ###############
                 use_moe: bool = False,
                 num_experts_per_tok: int = 2,
                 n_routed_experts: int = 4,
                 n_shared_experts: int = 1,
                 scoring_func: str = 'softmax',
                 norm_topk_prob: bool = True,
                 aux_loss_alpha: float = 0.1,
                 seq_aux: bool = True,
                 **kwconfig):
        super().__init__(**kwconfig)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_kv_heads = num_kv_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.norm_topk_prob = norm_topk_prob  # 是否归一化topk概率
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
    

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # [dim//2]
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() # (end, dim//2) 
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) # (end, dim)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

    

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # (bsz, seq_len, num_head, head_dim) * (seq_len, 1 , head_dim)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: ZModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.dropout = config.dropout
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn


    def forward(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.reshape(bsz, seq_len, self.num_attention_heads, self.head_dim)
        key = key.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value = value.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

        query, key = apply_rotary_pos_emb(query, key, position_embeddings[0][:seq_len], position_embeddings[1][:seq_len])

        # GQA: cache 存储压缩的 KV (num_kv_heads 维度)，节省显存
        # 先 transpose 再 concat，保持 [bsz, num_kv_heads, seq_len, head_dim]
        query = query.transpose(1, 2)  # [bsz, num_attention_heads, seq_len, head_dim]
        key = key.transpose(1, 2)      # [bsz, num_kv_heads, seq_len, head_dim]
        value = value.transpose(1, 2)  # [bsz, num_kv_heads, seq_len, head_dim]

        if past_key_values is not None and use_cache:
            # past_key_values 存储的是压缩的 KV (num_kv_heads 维度)
            key = torch.cat([past_key_values[0], key], dim=2)
            value = torch.cat([past_key_values[1], value], dim=2)

        # 缓存压缩的 KV，节省显存 (num_kv_heads 而不是 num_attention_heads)
        past_key_values = (key, value)

        key = key.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=1)
        value = value.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=1)
        
        kv_seq_len = key.shape[2]
        
        if self.flash:
            output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=True)
        else:
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.full((seq_len, kv_seq_len), float('-inf')), diagonal=1).to(attn_weights.device)
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf')) if attention_mask is not None else attn_weights
            # attn_weights = self.attn_dropout(attn_weights)
            score = F.softmax(attn_weights, dim=-1)
            output = torch.matmul(score, value)
            
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(output)
        return output, past_key_values


class RMSNorm(nn.Module):
    def __init__(self, config: ZModelConfig, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
    
    def _norm(self, hidden_states):
        rsqrt = torch.rsqrt(torch.mean(torch.pow(hidden_states, 2), dim=-1, keepdim=True) + self.epsilon)
        return hidden_states * rsqrt


    def forward(self, hidden_states) -> torch.Tensor:
        return self.weight * self._norm(hidden_states)


def silu_act(inputs: torch.Tensor):
    return inputs * torch.sigmoid(inputs)


class FeedForward(nn.Module):
    def __init__(self, config: ZModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.act_fn = silu_act
        # self.act_fn = ACT2FN[config.hidden_act]
        if not config.intermediate_size:
            config.intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = ((config.intermediate_size + 64 - 1) // 64) * 64
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)))

class MoEGate(nn.Module):
    def __init__(self, config: ZModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: ZModelConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.zeros_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
        


class ZModelLayer(nn.Module):
    def __init__(self, layer_id: int, config: ZModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.attention = Attention(config)
        self.feedforward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.layer_norm_atten = RMSNorm(config)
        self.layer_norm_ffn = RMSNorm(config)
    
    def forward(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm_atten(hidden_states)
        hidden_states, past_key_values = self.attention(hidden_states, position_embeddings, past_key_values, use_cache, attention_mask)
        hidden_states += residual
        hidden_states = self.feedforward(self.layer_norm_ffn(hidden_states)) + hidden_states
        aux_loss = self.feedforward.aux_loss if isinstance(self.feedforward, MOEFeedForward) else hidden_states.new_zeros(1).squeeze()
        return hidden_states, past_key_values, aux_loss



class ZModel(nn.Module):
    def __init__(self, config: ZModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout = nn.Dropout(config.dropout)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ZModelLayer(idx, config) for idx in range(config.num_hidden_layers)])
        self.layer_norm = RMSNorm(config)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self,
                input_ids: torch.Tensor,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs):

        _, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embeddings(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        cur_key_values = []
        total_aux_loss = None

        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, cur_key_value, aux_loss = layer(
                hidden_states,
                position_embeddings,
                past_key_values=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask     
            )
            cur_key_values.append(cur_key_value)
            total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss
        
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cur_key_values, total_aux_loss

class ZModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ZModelConfig
    
    def __init__(self, config: ZModelConfig = None):
        self.config = config or ZModelConfig()
        super().__init__(self.config)
        self.model = ZModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embeddings.weight = self.lm_head.weight
        
    def forward(self,
                input_ids: torch.Tensor,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None,
                **kwconfig):
        hidden_states, past_key_values, total_aux_loss = self.model(input_ids, 
                                                    past_key_values=past_key_values, 
                                                    use_cache=use_cache, 
                                                    attention_mask=attention_mask, 
                                                    **kwconfig)
        logits = self.lm_head(hidden_states)
        return MoeCausalLMOutputWithPast(
            hidden_states=hidden_states,
            logits=logits,
            past_key_values=past_key_values,
            aux_loss=total_aux_loss
        )


def sft_train(model: ZModelForCausalLM, train_data: List[Tuple[torch.Tensor, torch.Tensor]]):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(100):
        epoch_total_loss = 0.0
        for input_ids, target_ids in train_data:
            attention_mask = None
            outputs = model.forward(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = target_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shifted_logits.view(-1, model.config.vocab_size), shifted_labels.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_total_loss += loss.item()
        print(f"Epoch {epoch} loss: {epoch_total_loss / len(train_data)}")
    return model


@torch.no_grad()
def model_inference(model: ZModelForCausalLM, input_ids: torch.Tensor):
    logits = model.forward(input_ids)["logits"]
    token_ids = torch.argmax(logits, dim=-1)
    return token_ids


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = ZModelConfig(hidden_size=16, flash_attn=False, vocab_size=128)
    bs, seq = 16, 64
    input_ids = torch.randint(0, 128, (bs, seq)).to(device)
    model = ZModelForCausalLM(cfg)
    model.to(device)
    outputs = model.forward(input_ids)

    print(outputs)
    # print("rand_atten_input: ", rand_atten_input.shape)
    # print("all elements: ", rand_atten_input.numel())
    # print("atten_out: ", output.shape)
    # print("normed_hidden: ", normed_hidden.shape)

    train_data = [(torch.randint(0, 128, (bs, seq)), torch.randint(0, 128, (bs, seq)))]
    model = sft_train(model, train_data)
    token_ids = model_inference(model, input_ids)
    print(token_ids)

    