import torch.nn.functional as F
import torch
from torch import nn
import os
import copy
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

HEAD_NUM = int(os.environ.get("HEAD_NUM", 0))
BETA = float(os.environ.get("BETA", 0.5))
THRES = float(os.environ.get("THRES", 0))

import types
import torch
from torch import nn
import transformers.models.qwen2.modeling_qwen2 as qwen_module  # 假设是Qwen2架构





def patch_rope_for_registers(num_registers):
    """
    全局拦截 RoPE 计算，使其跳过前 num_registers 个 Token
    """
    if hasattr(qwen_module, "_is_patched_for_regs"):
        return

    old_apply_rotary = qwen_module.apply_rotary_pos_emb

    def patched_apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # 维度切分: [Batch, Heads, Seq_Len, Head_Dim]
        q_regs, q_text = q[:, :, :num_registers, :], q[:, :, num_registers:, :]
        k_regs, k_text = k[:, :, :num_registers, :], k[:, :, num_registers:, :]


        # 仅对文本部分旋转
        q_rot, k_rot = old_apply_rotary(q_text, k_text, cos, sin, position_ids, unsqueeze_dim)

        # 重新拼接
        return torch.cat([q_regs, q_rot], dim=2), torch.cat([k_regs, k_rot], dim=2)

    qwen_module.apply_rotary_pos_emb = patched_apply_rotary
    qwen_module._is_patched_for_regs = True
    print(f">>> RoPE Patch Applied: Bypassing first {num_registers} tokens.")


def act_language_register_without_position(model_component, num_registers=4, is_vision=False, is_train=True):
    print("enter act_language")
    if hasattr(model_component, "_is_redistributed"):
        return model_component

    # 1. 拦截底层 RoPE (仅针对 Language)
    if not is_vision:
        patch_rope_for_registers(num_registers)

    model_component._is_redistributed = True
    hidden_size = model_component.config.hidden_size
    reg_name = "v_regs" if is_vision else "t_regs"

    # 2. 初始化寄存器
    if not hasattr(model_component, reg_name):
        regs = nn.Parameter(torch.empty(1, num_registers, hidden_size))
        nn.init.trunc_normal_(regs, std=0.02)
        setattr(model_component, reg_name, regs)

    orig_forward = model_component.forward

    def forward(self, *args, **kwargs):
        # A. 提取并拼接寄存器
        if is_vision:
            # 视觉分支保持你原有的逻辑
            pixel_values = args[0] if args else kwargs.get('pixel_values')
            x, _ = self.embeddings(pixel_values)
        else:
            x = kwargs.get('inputs_embeds', None)
            if x is None:
                x = self.get_input_embeddings()(kwargs.get('input_ids'))

        batch_size, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        # 注入寄存器
        regs = getattr(self, reg_name).expand(batch_size, -1, -1)
        x = torch.cat([regs, x], dim=1)  # [B, N+L, D]

        # B. 准备参数 (Mask 和 Position IDs)
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            m = kwargs["attention_mask"]
            reg_m = torch.ones((batch_size, num_registers), dtype=m.dtype, device=device)
            kwargs["attention_mask"] = torch.cat([reg_m, m], dim=1)

        if not is_vision:
            p = kwargs.get("position_ids", None)
            if p is None:
                p = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            # 寄存器占位 position 0
            reg_p = torch.zeros((batch_size, num_registers), dtype=p.dtype, device=device)
            kwargs["position_ids"] = torch.cat([reg_p, p], dim=1)

            # C. 调用原始 forward
            # 此时内部调用 apply_rotary_pos_emb 会自动识别并跳过寄存器
            kwargs.pop('inputs_embeds', None)
            outputs = orig_forward(inputs_embeds=x, input_ids=None, **kwargs)

            # D. 后处理：剥离寄存器
            # 修改 last_hidden_state
            outputs.last_hidden_state = outputs.last_hidden_state[:, num_registers:, :]

            # 修改 attentions (如果是元组，则逐层切分)
            if outputs.attentions is not None:
                new_attns = []
                for layer_attn in outputs.attentions:
                    # [B, H, N+L, N+L] -> [B, H, L, L] 剥离寄存器的 Query 和 Key
                    # 1. 寄存器作为 Query：它们在看什么？
                    # [B, H, num_regs, num_regs] -> 寄存器自关注
                    # print(
                    #     f"Self-attention among registers: {layer_attn[:, :, :num_registers, :num_registers].sum(dim=-1).mean()}")
                    # # [B, H, num_regs, L] -> 寄存器从文本中提取了多少信息
                    # print(
                    #     f"Registers attending to text: {layer_attn[:, :, :num_registers, num_registers:].sum(dim=-1).mean()}")
                    # # 2. 文本作为 Query：寄存器吸收了多少注意力？ (这是判断 Sink 的关键)
                    # # [B, H, L, num_regs] -> 文本对寄存器的注意力
                    # text_to_reg_attn = layer_attn[:, :, num_registers:, :num_registers].sum(dim=-1)
                    # print(f"Text tokens attending to registers (Sink Absorption): {text_to_reg_attn.mean()}")
                    # # 3. 最后一个词的分布对比
                    # # 最后一个词看寄存器 vs 看其他文本
                    # last_token_to_reg = layer_attn[:, :, -1, :num_registers].sum(dim=-1)
                    # last_token_to_text = layer_attn[:, :, -1, num_registers:].sum(dim=-1)
                    # print(
                    #     f"Last token attention: To Registers={last_token_to_reg.mean():.4f}, To Text={last_token_to_text.mean():.4f}")

                    new_attns.append(layer_attn[:, :, num_registers:, num_registers:])
                outputs.attentions = tuple(new_attns)

            return outputs

    model_component.forward = types.MethodType(forward, model_component)
    return model_component

def internvl_vision_atten_forward_hook(self, hidden_states, attention_mask=None, **kwargs):
    batch_size, seq_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # No upcasting of the attention weights to float32 in this implementation
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    dropout = 0.0 if not self.training else self.attention_dropout
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()


    # 5.--------redistribution MODIFY PART ---------------
    attn_weights = atten_process_eval(attn_weights)
    # -------- -------- -------- -------- --------

    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

    output = self.projection_layer(attn_output)
    output = self.projection_dropout(output)

    return output, attn_weights


def internvl_language_atten_forward_hook(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        layer_idx=None,
        **kwargs
):
    # 1. 获取基础参数
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len, _ = hidden_states.shape
    # 从 self (attention 层) 的父节点或模型引用中获取 t_regs
    # 建议初始化使用: nn.init.ones_(t_regs) 或 trunc_normal_(t_regs, std=0.5)
    t_regs = getattr(self, "t_regs", None)
    num_regs = t_regs.shape[1] if t_regs is not None else 0

    # 2. 生成 Q, K, V
    # Query 只针对原始文本生成的 hidden_states [B, L, D]
    query_states = self.q_proj(hidden_states)

    # Key 和 Value 需要包含 Register 信息
    if num_regs > 0:
        # 这种方案下，我们不在 internal_hidden 里拼，而是直接拼 Projection 后的结果
        # 或者为了让 Regs 经过投影，构造一个临时的拼接输入
        reg_input_shape = t_regs.shape[:-1]
        reg_hidden_shape = (*reg_input_shape, -1, self.head_dim)

        reg_k = self.k_proj(t_regs).expand(bsz, -1, -1)
        reg_v = self.v_proj(t_regs).expand(bsz, -1, -1)

        k_text = self.k_proj(hidden_states)
        v_text = self.v_proj(hidden_states)

        # 后面会统一进行 Norm 和 View
    else:
        k_text = self.k_proj(hidden_states)
        v_text = self.v_proj(hidden_states)

    # 3. Norm 和 View 变换 (关键：先 Norm 再 Reshape)

    #     hidden_shape = (*input_shape, -1, self.head_dim)
    #     # --- 3. QKV  ---
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)

    k_text = self.k_norm(k_text.view(hidden_shape)).transpose(1, 2)
    v_text = v_text.view(hidden_shape).transpose(1, 2)

    if num_regs > 0:
        k_reg = self.k_norm(reg_k.view(reg_hidden_shape)).transpose(1, 2)
        v_reg = reg_v.view(reg_hidden_shape).transpose(1, 2)

    # 4. 应用 RoPE (仅针对文本部分)
    # 这种方案 Q 始终是原始长度，位置编码索引 100% 正确
    cos, sin = position_embeddings
    query_states, k_text = apply_rotary_pos_emb(query_states, k_text, cos, sin)

    # 5. KV 拼接 (入场)
    if num_regs > 0:
        # 将不带位置信息的 Register 拼在文本 Key 的前面
        key_states = torch.cat([k_reg, k_text], dim=2)
        value_states = torch.cat([v_reg, v_text], dim=2)
    else:
        key_states, value_states = k_text, v_text

    # 6. 组对齐 (GQA/MQA)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 7. Attention Mask 适配 (手动放行 Register)
    if attention_mask is not None and num_regs > 0:
        # 构造一个全 0 (可见) 的 mask 块，维度为 [B, 1, L, num_regs]
        reg_mask = torch.zeros((bsz, 1, q_len, num_regs),
                               device=attention_mask.device,
                               dtype=attention_mask.dtype)
        # 拼接到原始因果掩码的前面
        curr_mask = torch.cat([reg_mask, attention_mask], dim=-1)
    else:
        curr_mask = attention_mask

    # 8. 计算注意力权重
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if curr_mask is not None:
        attn_weights = attn_weights + curr_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # 执行你的重分布/分析逻辑 (此时 attn_weights 最后一维包含 Register)
    attn_weights = atten_process_eval(attn_weights)

    # 9. 输出计算 (由于 Q 是 L，输出自然是 L)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    print(f"Register Attention: {attn_weights[0, 0, :, :num_regs].sum().item()}")
    # 10. 返回结果 (不需要任何截断，外层残差连接直接匹配)
    return attn_output, attn_weights


# def internvl_language_atten_forward_hook(
#         self,
#         hidden_states,
#         position_embeddings=None,  # 必须接收这个
#         attention_mask=None,
#         past_key_values=None,
#         cache_position=None,
#         layer_idx=None,
#         **kwargs
# ):
#
#     t_regs = getattr(self, "t_regs", None)
#     num_regs = t_regs.shape[1] if t_regs is not None else 0
#     bsz, q_len_orig, _ = hidden_states.shape
#
#     if t_regs is not None:
#         regs = t_regs.expand(bsz, -1, -1).to(hidden_states.dtype)
#         # 内部 hidden_states 变为 [B, N+L, D]
#         internal_hidden = torch.cat([regs, hidden_states], dim=1)
#     else:
#         internal_hidden = hidden_states
#     input_shape = internal_hidden.shape[:-1]
#     hidden_shape = (*input_shape, -1, self.head_dim)
#     # --- 3. QKV  ---
#     query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#     key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#     value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#
#     # --- 4. RoPE processing (remove Register to make it without position information) ---
#     cos, sin = position_embeddings
#
#     # if num_regs > 0:
#     #     # split register token and original q and k
#     #     q_regs, q_texts = query_states[:, :, :num_regs, :], query_states[:, :, num_regs:, :]
#     #     k_regs, k_texts = key_states[:, :, :num_regs, :], key_states[:, :, num_regs:, :]
#     #     # apply RoPE to the original q and k
#     #     q_texts, k_texts = apply_rotary_pos_emb(q_texts, k_texts, cos, sin)
#     #     # # recombine
#     #     # query_states = torch.cat([q_regs, q_texts], dim=2)
#     #     # key_states = torch.cat([k_regs, k_texts], dim=2)
#     # else:
#     #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#     # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#     # --- 5. 组对齐 (GQA/MQA) ---
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)
#
#     # --- 6. Attention Mask 适配 ---
#     # # 原始 mask 是 [B, 1, L, L]，现在需要变成 [B, 1, N+L, N+L]
#     # if attention_mask is not None:
#     #     # 创建一个全 0 (可见) 的 mask 给 register 部分
#     #     # 形状对齐到 [B, 1, N+L, N]
#     #     reg_mask_col = torch.zeros((bsz, 1, query_states.shape[2], num_regs),
#     #                                device=attention_mask.device, dtype=attention_mask.dtype)
#     #     # 拼接到 Key 维度上
#     #     curr_mask = torch.cat([reg_mask_col, attention_mask], dim=-1)
#     #
#     # else:
#     #     curr_mask = None
#
#     # --- 修正后的 Mask 处理逻辑 ---
#     if attention_mask is not None:
#         # 1. 识别维度: attention_mask 形状通常是 [bsz, 1, q_len_orig, kv_len_orig]
#         # 我们现在的 q_len 是 q_len_orig (+ num_regs 如果不是最后一层)
#         # 我们现在的 kv_len 是 kv_len_orig + num_regs
#
#         # 2. 构造一个全 0 (代表可见) 的区域给 Register 这一列
#         # 形状为 [bsz, 1, query_states.shape[2], num_regs]
#         reg_mask_cols = torch.zeros(
#             (bsz, 1, query_states.shape[2], num_regs),
#             device=attention_mask.device,
#             dtype=attention_mask.dtype
#         )
#
#         # 3. 处理 Query 维度的对齐
#         # 如果是最后一层，query_states 只有原始长度 L
#         if query_states.shape[2] == q_len_orig:
#             # 直接在横向拼接 Register 的列掩码
#             curr_mask = torch.cat([reg_mask_cols, attention_mask], dim=-1)
#         else:
#             # 如果是中间层，Query 也有 N 个，我们需要补全 Register 自己的 Mask 行
#             # Register 之间通常也是互见的，所以补一个全 0 的行块
#             reg_mask_rows = torch.zeros(
#                 (bsz, 1, num_regs, key_states.shape[2]),
#                 device=attention_mask.device,
#                 dtype=attention_mask.dtype
#             )
#             # 先在 Key 轴拼 Register 列
#             extended_mask = torch.cat([reg_mask_cols[:, :, num_regs:, :], attention_mask], dim=-1)
#             # 再在 Query 轴拼 Register 行
#             curr_mask = torch.cat([reg_mask_rows, extended_mask], dim=2)
#     else:
#         curr_mask = None
#
#
#     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
#     # if curr_mask is not None:
#     #     attn_weights = attn_weights + curr_mask
#
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#
#     ###### redistribution
#     # attn_weights = atten_process_eval(attn_weights)
#
#     attn_output = torch.matmul(attn_weights, value_states)  # [B, H, N+L, D_head]
#
#     attn_output = attn_output.transpose(1, 2).contiguous()
#     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#
#     attn_output = self.o_proj(attn_output)
#     # 检查第一个 head 的第一行（第一个 text token）对 register 的关注度
#     print(f"Sample Attention to Register: {attn_weights[ :, :, num_regs, :num_regs].sum(-1)}")
#     if num_regs > 0:
#         attn_output = attn_output[:, num_regs:, :]
#
#     return attn_output, attn_weights
#
#


def atten_process_eval(attention_map,  threshold=0.2, index=None):
    # attention_map: [B, H, S, S]
    B, H, S, _ = attention_map.shape
    device = attention_map.device

    # 1. 检测 Sink (保持之前的逻辑)
    avg_head_attn = attention_map.mean(dim=1)  # [B, S, S]
    col_sums = avg_head_attn.sum(dim=1)  # [B, S]
    contribution_ratio = col_sums / S  # [B, S]

    for b in range(B):
        first_val = contribution_ratio[b, 0].item()
        last_val = contribution_ratio[b, -1].item()

        print(f"\n" + "=" * 30)
        print(f"BATCH {b} MONITOR")
        print(f"Boundary - First [0]: {first_val:.4f} | Last [-1]: {last_val:.4f}")

        # 找到该 batch 下所有的 sink 索引
        batch_sink_indices = torch.where(contribution_ratio[b] > threshold)[0]

        if len(batch_sink_indices) > 0:
            print(f"Detected Sinks (Total: {len(batch_sink_indices)}):")
            for s_idx in batch_sink_indices:
                s_idx_item = s_idx.item()
                # 标记一下特殊的词
                tag = ""
                if s_idx_item == 0:
                    tag = "[START/SINK]"
                elif s_idx_item == S - 1:
                    tag = "[END]"

                val = contribution_ratio[b, s_idx_item].item()
                print(f"  - Index {s_idx_item:4d}: {val:.4f} {tag}")
        else:
            print("No Sinks detected above threshold.")

    # sink_mask shape: [B, 1, 1, S] -> 方便后续在 H, Query_S 维度上广播
    sink_mask = (contribution_ratio > threshold).float().view(B, 1, 1, S)

    # 2. 计算要从 Sink 中提取的权重
    # head 是 [B, H, S, S], sink_mask 在最后维度进行筛选
    # sink_weights: [B, H, S, S] (只保留 Sink 列的数值)
    sink_weights = attention_map * sink_mask

    # available_weights: [B, H, S] (每一行被提取出的总能量)
    available_weights = sink_weights.sum(dim=-1) * (1 - BETA)

    # 3. 对原始 Map 中的 Sink 列应用 Beta 惩罚
    # penalty_map: [B, 1, 1, S]
    penalty_map = 1.0 - (sink_mask * (1 - BETA))
    attention_map = attention_map * penalty_map

    # 4. 准备再分配比例 (Redistribution Ratios)
    # pool: [B, H, S, S] (排除掉 Sink 列，只在非 Sink 列分配)
    pool = attention_map.detach() * (1.0 - sink_mask)
    row_sums = pool.sum(dim=-1, keepdim=True)  # [B, H, S, 1]

    # ratios: [B, H, S, S]
    ratios = pool / (row_sums + 1e-9)

    # 5. 将提取的能量按比例加回到每一行
    # available_weights.unsqueeze(-1) 是 [B, H, S, 1]
    # ratios 是 [B, H, S, S]
    attention_map = attention_map + (available_weights.unsqueeze(-1) * ratios)

    return attention_map

def atten_process_cal(attention_map):
    beta = BETA
    threshod = THRES

    modified_head = attention_map[0][HEAD_NUM]
    device = modified_head.device

    shape = modified_head.shape[0]
    indices = torch.nonzero(torch.where(torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device) > threshod, 1, 0))
    indices = indices[1:]
    copied_attention_map = copy.deepcopy(modified_head.detach())
    available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
    modified_head[:, indices] *= beta
    copied_attention_map[1:, indices] *= 0
    ratios = copied_attention_map / torch.sum(copied_attention_map,  dim=1, keepdim=True).to(copied_attention_map.dtype)
    modified_head = modified_head + available_weights * ratios
    modified_head[0, 0] = 1

    attention_map[0][HEAD_NUM] = modified_head

    return attention_map
