import re

def gsm8k_reward_fn(response: str, ground_truth: str) -> dict:
    """
    GSM8K 的 rule-based reward 函数。
    奖励组成：答案正确 1.0 分 + 格式正确 0.1 分。
    返回 dict，训练循环从中提取 'reward' 标量。
    """
    # 从 ground_truth 中提取标准答案（GSM8K 格式：#### 42）
    gt_match = re.search(r'####\s*(\-?[\d,]+)', ground_truth)
    gt_answer = gt_match.group(1).replace(',', '') if gt_match else ''

    # 从模型 response 中提取答案（支持 <answer> 和 #### 两种格式）
    resp_match = (re.search(r'<answer>\s*(\-?[\d,]+)', response) or
                  re.search(r'####\s*(\-?[\d,]+)', response))
    resp_answer = resp_match.group(1).replace(',', '') if resp_match else ''

    # 答案是否正确（精确字符串匹配）
    answer_correct = float(gt_answer == resp_answer)
    # 是否遵循了 <think>...<answer>... 的格式约定
    has_format = float('<think>' in response and '<answer>' in response)

    return {
        'reward': answer_correct + 0.1 * has_format,  # 总 reward = 答案 + 格式 bonus
        'answer_reward': answer_correct,                # 用于 logging
        'format_reward': has_format                     # 用于 logging
    }