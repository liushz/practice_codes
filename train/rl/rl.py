import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, List
import re
from enum import Enum
from functools import wraps
from transformers import AutoModelForCausalLM

class Dispatch(Enum):
    """分发模式枚举，决定 WorkerGroup 如何将数据分发给各 Worker。"""
    ONE_TO_ALL = "one_to_all"       # 广播模式：将完整输入复制发给每个 Worker
    DP_COMPUTE_PROTO = "dp_proto"   # 数据并行模式：将 DataProto 按 batch 维切分后分发

# 用于在函数对象上标记注册信息的魔法属性名
MAGIC_ATTR = "__mini_verl_register__"

def register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True):
    """
    核心装饰器：声明 Worker 方法的分发行为。
    被装饰的方法会在函数对象上挂载一个 MAGIC_ATTR 属性，
    WorkerGroup 初始化时会扫描这个属性来自动生成分发方法。

    参数：
      dispatch_mode: 分发模式（ONE_TO_ALL 或 DP_COMPUTE_PROTO）
      blocking: 是否同步等待所有 Worker 完成（mini 版暂时都用 True）
    """
    def decorator(func):
        @wraps(func)                    # 保留原函数的 __name__、__doc__ 等元信息
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        # 在包装后的函数上挂载注册元信息，供 WorkerGroup 读取
        setattr(inner, MAGIC_ATTR, {
            "dispatch_mode": dispatch_mode,
            "blocking": blocking
        })
        return inner
    return decorator


@dataclass
class DataProto:
    """
    Mini 版 VERL DataProto —— 统一的 tensor 数据容器。
    所有模块（Actor、Critic、Reward、Rollout）之间的数据交换都通过它完成。
    核心结构：batch (tensor 字典) + meta_info (非 tensor 元数据)
    """
    batch: Dict[str, torch.Tensor] = field(default_factory=dict)   # 存放各种 tensor：input_ids, rewards, logprobs 等
    meta_info: Dict[str, Any] = field(default_factory=dict)        # 存放非 tensor 信息：如 tokenizer config、任务描述等

    def to(self, device):
        """将所有 tensor 搬运到指定设备（CPU/GPU），meta_info 保持不变。"""
        return DataProto(
            batch={k: v.to(device) for k, v in self.batch.items()},
            meta_info=self.meta_info
        )

    def chunk(self, n: int) -> List['DataProto']:
        """
        按 batch 维度（dim=0）将数据均匀切分为 n 份。
        用于分布式场景中将数据分发给 n 个 Worker。
        注意：meta_info 是共享引用，不做深拷贝。
        """
        chunks = [{} for _ in range(n)]           # 初始化 n 个空字典
        for k, v in self.batch.items():           # 遍历每个 tensor 字段
            for i, c in enumerate(v.chunk(n, dim=0)):  # 沿 batch 维切分
                chunks[i][k] = c
        return [DataProto(batch=c, meta_info=self.meta_info) for c in chunks]

    @staticmethod
    def concat(protos: List['DataProto']) -> 'DataProto':
        """
        将多个 DataProto 沿 batch 维度合并为一个。
        用于收集各 Worker 的结果后拼回完整 batch。
        meta_info 采用合并策略（后者覆盖前者）。
        """
        keys = protos[0].batch.keys()
        merged = {k: torch.cat([p.batch[k] for p in protos], dim=0) for k in keys}
        # 合并所有 proto 的 meta_info，后面的值会覆盖前面的
        combined_meta = {}
        for p in protos:
            combined_meta.update(p.meta_info)
        return DataProto(batch=merged, meta_info=combined_meta)

    def __getitem__(self, key):
        """支持字典式读取：proto['input_ids']"""
        return self.batch[key]

    def __setitem__(self, key, value):
        """支持字典式写入：proto['rewards'] = reward_tensor"""
        self.batch[key] = value
        

class ActorWorker:
    """
    Actor Worker：负责模型的 rollout（生成）和 policy 更新。
    每个 Worker 实例绑定一张 GPU，WorkerGroup 会创建多个实例实现数据并行。
    """
    def __init__(self, model_path, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path).to(device)      # 加载模型并放到指定 GPU
        self.device = device

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)  # 数据并行：输入会被自动切分
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Rollout：根据 prompt 生成 response 序列。"""
        prompts = prompts.to(self.device)               # 将分到的 chunk 搬到本 Worker 的 GPU
        input_ids = prompts['input_ids']
        outputs = self.model.generate(input_ids, max_new_tokens=2048)
        return DataProto(batch={'sequences': outputs})  # 封装结果，WorkerGroup 会自动 concat

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)  # 同样数据并行
    def update_actor(self, data: DataProto) -> DataProto:
        """Policy 更新：用 GRPO/PPO loss 更新模型参数。"""
        



def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    将 prompt + response 文本对打包为模型输入格式。
    返回 input_ids, labels（错位一位）, attention_mask, response_mask。

    布局示意（以单条样本为例）：
    原始 tokens:  [p1, p2, p3, o1, o2, o3, <eos>, <pad>, <pad>]
    input_ids:    [p1, p2, p3, o1, o2, o3, <eos>, <pad>]   # 去掉最后一个
    labels:       [p2, p3, o1, o2, o3, <eos>, <pad>, <pad>] # 去掉第一个（next-token prediction）
    attn_mask:    [1,  1,  1,  1,  1,  1,  1,    0]        # padding 位置为 0
    resp_mask:    [0,  0,  1,  1,  1,  1,  0,    0]        # 只标记 response 部分
    """
    prompt_tokens = tokenizer(prompt_strs)['input_ids']     # 分别 tokenize prompt
    output_tokens = tokenizer(output_strs)['input_ids']     # 和 response
    batch_sz = len(prompt_tokens)
    prompt_and_output_lens = [len(p) + len(o)
                              for p, o in zip(prompt_tokens, output_tokens)]
    padded_len = max(prompt_and_output_lens)                # 对齐到最长序列

    # 预分配张量（padded_len - 1 是因为 input/label 错位后少一个位置）
    input_ids = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    labels = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    attention_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.long)   # 0 = 忽略 padding
    response_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.float)   # float 方便后续做 loss 加权

    for i, (p_toks, o_toks) in enumerate(zip(prompt_tokens, output_tokens)):
        concat = torch.tensor(p_toks + o_toks)              # 拼接 prompt + response tokens
        concat_len = len(concat)
        # 右侧用 eos_token_id 填充到 padded_len
        padded = F.pad(concat, (0, padded_len - concat_len),
                       'constant', tokenizer.eos_token_id)
        input_ids[i] = padded[:-1]                          # 模型输入：去掉最后一个 token
        labels[i] = padded[1:]                              # 预测目标：去掉第一个 token（next-token prediction）
        attention_mask[i, :concat_len - 1] = 1              # 有效 token 位置设为 1，padding 保持 0
        # response 在 labels 中的起止位置
        o_start = len(p_toks) - 1                           # labels 中 response 开始的 index
        o_end = concat_len - 1                              # labels 中 response 结束的 index（不含）
        response_mask[i, o_start:o_end] = 1.0               # 标记 response token
    return {'input_ids': input_ids, 'labels': labels,
            'attention_mask': attention_mask, 'response_mask': response_mask}


def get_response_log_probs(model, input_ids, labels, attention_mask):
    """
    计算模型在每个 token 位置上对 labels 的 log probability。
    返回 [B, T] 的 per-token log prob，后续配合 response_mask 使用。
    """
    logits = model(input_ids, attention_mask=attention_mask).logits  # [B, T, V] 前向传播
    log_probs = F.log_softmax(logits, dim=-1)                       # 转为 log 概率分布
    # 从词表维度上 gather 出 labels 对应位置的 log prob
    log_probs = torch.gather(log_probs, dim=-1,
                             index=labels.unsqueeze(-1)).squeeze(-1) # [B, T]
    return log_probs

def compute_group_advantage(rewards: torch.Tensor, group_size: int,
                            eps: float = 1e-8) -> torch.Tensor:
    """
    计算 GRPO 的 group-level advantage。
    rewards: [N], N = num_prompts * group_size，每个 response 一个标量 reward。
    核心思想：同一个 prompt 的 G 个 response 在组内做 z-score 归一化，
    好的 response 得到正 advantage，差的得到负 advantage。
    """
    assert group_size > 1, "GRPO requires group_size > 1 for meaningful advantages"
    grouped = rewards.reshape(-1, group_size)          # [num_prompts, G]
    mean = grouped.mean(dim=-1, keepdim=True)          # 组内均值 [num_prompts, 1]
    std = grouped.std(dim=-1, keepdim=True)             # 组内标准差 [num_prompts, 1]
    advantage = (grouped - mean) / (std + eps)          # z-score 归一化，eps 防除零
    return advantage.flatten()                          # 展平回 [N]

def grpo_clip_loss(logprobs: torch.Tensor, old_logprobs: torch.Tensor,
                   advantages: torch.Tensor, response_mask: torch.Tensor,
                   clip_eps: float = 0.2, beta: float = 0.01,
                   ref_logprobs: torch.Tensor = None) -> torch.Tensor:
    """
    GRPO 的 per-token clipped 目标函数，带可选 KL 惩罚。
    logprobs:      当前 policy 的 per-token log prob  [B, T]
    old_logprobs:  rollout 时冻结的 log prob（detached）[B, T]
    advantages:    每个 sample 的 group advantage      [B]
    response_mask: 标记哪些 token 是 response 部分     [B, T]
    clip_eps:      PPO-clip 范围，默认 0.2
    beta:          KL 惩罚系数
    ref_logprobs:  参考模型的 log prob（用于 KL 约束）  [B, T]
    """
    # 重要性采样比率 r(θ) = π_θ(a|s) / π_old(a|s)
    ratio = torch.exp(logprobs - old_logprobs)          # [B, T]
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)  # 截断 ratio

    # advantage 从 [B] 扩展到 [B, 1]，广播到每个 token
    adv = advantages.unsqueeze(-1)

    # PPO-clip 目标：取 min(ratio * A, clip(ratio) * A) 的负值作为 loss
    pg_loss = -torch.min(ratio * adv, clipped * adv)

    # 可选：KL(π_θ || π_ref) ≈ logprobs - ref_logprobs，防止 policy 偏离太远
    kl_loss = 0.0
    if ref_logprobs is not None:
        kl_loss = beta * (logprobs - ref_logprobs)

    # 只在 response token 上计算 loss，忽略 prompt 和 padding
    loss = (pg_loss + kl_loss) * response_mask
    return loss.sum() / response_mask.sum()             # 按有效 token 数归一化



class MiniGRPOTrainer:
    """
    GRPO 训练器骨架。
    整体流程：rollout → reward → advantage → policy update（多轮 mini-epoch）。
    """
    def __init__(self, model, ref_model, tokenizer, reward_fn,
                 group_size=8, lr=1e-6, clip_eps=0.2):
        self.model = model                  # 可训练的 policy 模型
        self.ref_model = ref_model          # 冻结的参考模型副本（用于 KL 约束）
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn          # reward 函数，返回 dict（含 'reward' key）
        self.group_size = group_size        # 每个 prompt 采样的 response 数量 G
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps            # PPO-clip 范围

    def fit(self, dataloader, epochs=3):
        for epoch in range(epochs):
            for batch in dataloader:
                prompts = batch['prompts']

                # ---- Step 1: Rollout ----
                # 对每个 prompt 用当前 policy 采样 G 个 response，同时记录 log prob
                responses, old_logprobs = self.rollout(prompts)
                # 必须 detach！old_logprobs 在后续作为常量使用，不参与梯度计算
                old_logprobs = old_logprobs.detach()

                # ---- Step 2: Reward ----
                # 调用 reward_fn 得到 dict，提取 'reward' 标量组装为 tensor
                reward_dicts = [self.reward_fn(r, a)
                                for r, a in zip(responses, batch['answers'])]
                rewards = torch.tensor([d['reward'] for d in reward_dicts])

                # ---- Step 3: Group Advantage ----
                # 在同一 prompt 的 G 个 response 内做 z-score 归一化
                advantages = compute_group_advantage(rewards, self.group_size)

                # ---- Step 4: Ref Log Probs（用于 KL 惩罚）----
                # 参考模型不更新，用 no_grad 节省显存
                with torch.no_grad():
                    ref_logprobs = self.compute_logprobs(self.ref_model, ...)

                # ---- Step 5: Policy Update（多轮 mini-epoch）----
                # 用同一批 rollout 数据多次更新 policy（类似 PPO 的多 epoch）
                for _ in range(4):
                    curr_logprobs = self.compute_logprobs(self.model, ...)
                    loss = grpo_clip_loss(
                        curr_logprobs, old_logprobs, advantages,
                        response_mask, self.clip_eps,
                        ref_logprobs=ref_logprobs
                    )
                    self.optimizer.zero_grad()   # 先清梯度
                    loss.backward()              # 反向传播
                    self.optimizer.step()         # 更新参数