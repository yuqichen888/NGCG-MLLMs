from torch.optim import AdamW
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from typing import Dict
from transformers import PreTrainedModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.arguments import ModelArguments, TrainingArguments, DataArguments
from src.model_utils import get_backbone_name, QWEN2_VL, QWEN2_5_VL, \
    backbone2model, SMOLVLM, IDEFICS3, INTERNVL, PHI3V
import pytorch_lightning as pl
import torch
from transformers import AutoConfig
from torch import nn, Tensor
import os
import json
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import gc
from math import radians, sin, cos, sqrt, atan2
import re
from transformers import BitsAndBytesConfig
from src.attn_hock import internvl_language_atten_forward_hook,internvl_vision_atten_forward_hook,act_language_register_without_position
import types
from functools import partial


@rank_zero_only
def my_print(*args, **kwargs):
    print(*args, **kwargs)


def check_matrix(mat, name):
    if np.any(np.isinf(mat)):
        print(f"WARNING: {name} contains inf!")
    if np.any(np.isnan(mat)):
        print(f"WARNING: {name} contains NaN!")
    if np.max(np.abs(mat)) > 1e10:
        print(f"WARNING: {name} has extreme values (max={np.max(mat)})")




def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def extract_lat_lon(filename):
    base_name = os.path.basename(filename)
    match = re.match(r"([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)", base_name)
    lat, lng = float(match.group(1)),float(match.group(2))
    return lat, lng

def cvg_cal_score_unified(similarities, is_panorama, image_locations, text_locations, m=100, mode='text2image'):
    """
    Params:
     similarities : np.ndarray with the shape of [num_images, num_texts]
     is_panorama : list[bool]
     image_locations : list[[lat, lon]]
     text_locations : list[[lat, lon]]
     m : int (the number of candidate in the geographical location)
     mode : str ('text2image' or 'image2text')
    """

    # --------- Step1: normalizing inputs ----------
    if isinstance(image_locations, torch.Tensor):
        image_locations = torch.round(image_locations * 1e8) / 1e8
        text_locations = torch.round(text_locations * 1e8) / 1e8
        image_locations = image_locations.tolist()
        text_locations = text_locations.tolist()
    elif isinstance(image_locations, list) and isinstance(image_locations[0], float):
        image_locations = [round(loc, 8) for loc in image_locations]
        text_locations = [round(loc, 8) for loc in text_locations]

    # --------- Step2: init metrix ----------
    def init_metrics():
        return {"R@1": 0, "R@5": 0, "R@10": 0,
                "L@50": 0, "L@100": 0, "L@150": 0,
                "Deviation": 0, "Count": 0}

    overall_metrics = init_metrics()
    panorama_metrics = init_metrics()
    single_view_metrics = init_metrics()
    top_k_records = []

    # task mode
    if mode == 'text2image':
        query_locs = text_locations
        target_locs = image_locations
        query_label = "Text"
        target_label = "Image"
        print("Evaluating Text → Image Retrieval...")
    elif mode == 'image2text':
        query_locs = image_locations
        target_locs = text_locations
        query_label = "Image"
        target_label = "Text"
        print("Evaluating Image → Text Retrieval...")
    else:
        raise ValueError("mode must be 'text2image' or 'image2text'")

    total_samples = len(query_locs)

    # --------- Step3: iterate samples ----------
    for i in range(total_samples):
        original_lat, original_lon = query_locs[i]
        is_pano = is_panorama[i]

        # computing distance
        distances = []
        for j, loc in enumerate(target_locs):
            lat, lon = loc
            distance = haversine(original_lat, original_lon, lat, lon)
            distances.append((distance, j))
        distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in distances[:m]]
        # take similarity in the matrix
        if mode == 'text2image':
            sim = similarities[closest_indices, i]
        else:
            sim = similarities[i, closest_indices]

        sorted_indices = np.argsort(-sim)
        best_matches = [target_locs[closest_indices[idx]] for idx in sorted_indices]
        original_loc = query_locs[i]

        # comparision
        best_match_t = torch.tensor(best_matches[0], dtype=torch.float32)
        original_loc_t = torch.tensor(original_loc, dtype=torch.float32)
        best_match_rounded = torch.round(best_match_t * 1e7) / 1e7
        original_loc_rounded = torch.round(original_loc_t * 1e7) / 1e7
        label_round = torch.equal(best_match_rounded, original_loc_rounded)

        # computing deviation
        if label_round:
            deviation = 0
        else:
            lat_match, lon_match = best_matches[0]
            deviation = haversine(original_lat, original_lon, lat_match, lon_match)

        l_50 = deviation <= 0.05
        l_100 = deviation <= 0.1
        l_150 = deviation <= 0.15

        for metrics in [overall_metrics, panorama_metrics if is_pano else single_view_metrics]:
            metrics["Count"] += 1
            metrics["Deviation"] += deviation
            metrics["L@50"] += int(l_50)
            metrics["L@100"] += int(l_100)
            metrics["L@150"] += int(l_150)

            def to_rounded_tuple(x, scale=1e7):
                if isinstance(x, torch.Tensor):
                    arr = x.detach().cpu().numpy()
                else:
                    arr = np.array(x, dtype=np.float32)
                arr = np.round(arr * scale) / scale
                return tuple(arr.tolist())

            ori_tuple = to_rounded_tuple(original_loc)
            best_rounded = [to_rounded_tuple(bm) for bm in best_matches]
            metrics["R@1"] += int(ori_tuple == best_rounded[0])
            metrics["R@5"] += int(ori_tuple in best_rounded[:5])
            metrics["R@10"] += int(ori_tuple in best_rounded[:10])

        top_k_records.append({
            f'original_{query_label.lower()}': tuple(original_loc_rounded.tolist()),
            f'top_5_{target_label.lower()}s': [bm for bm in best_matches[:5]],
            'label': label_round
        })

    def calculate_average(metrics):
        count = metrics["Count"]
        if count == 0:
            return metrics
        return {
            "R@1": (metrics["R@1"] / count) * 100,
            "R@5": (metrics["R@5"] / count) * 100,
            "R@10": (metrics["R@10"] / count) * 100,
            "L@50": (metrics["L@50"] / count) * 100,
            "L@100": (metrics["L@100"] / count) * 100,
            "L@150": (metrics["L@150"] / count) * 100,
            "Deviation": metrics["Deviation"] / count,
            "Count": count
        }

    overall_metrics = calculate_average(overall_metrics)
    panorama_metrics = calculate_average(panorama_metrics)
    single_view_metrics = calculate_average(single_view_metrics)

    print(f"\nOverall {query_label}→{target_label} Results:")
    print(f"R@1: {overall_metrics['R@1']:.2f}%")
    print(f"R@5: {overall_metrics['R@5']:.2f}%")
    print(f"R@10: {overall_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {overall_metrics['Deviation']:.2f} km")
    print(f"L@50: {overall_metrics['L@50']:.2f}%")
    print(f"L@100: {overall_metrics['L@100']:.2f}%")
    print(f"L@150: {overall_metrics['L@150']:.2f}%")

    print(f"\nPanorama Results ({query_label}→{target_label}):")
    print(f"R@1: {panorama_metrics['R@1']:.2f}%")
    print(f"R@5: {panorama_metrics['R@5']:.2f}%")
    print(f"R@10: {panorama_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {panorama_metrics['Deviation']:.2f} km")
    print(f"L@50: {panorama_metrics['L@50']:.2f}%")
    print(f"L@100: {panorama_metrics['L@100']:.2f}%")
    print(f"L@150: {panorama_metrics['L@150']:.2f}%")
    print(f"Num: {panorama_metrics['Count']}")

    print(f"\nSingle-View Results ({query_label}→{target_label}):")
    print(f"R@1: {single_view_metrics['R@1']:.2f}%")
    print(f"R@5: {single_view_metrics['R@5']:.2f}%")
    print(f"R@10: {single_view_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {single_view_metrics['Deviation']:.2f} km")
    print(f"L@50: {single_view_metrics['L@50']:.2f}%")
    print(f"L@100: {single_view_metrics['L@100']:.2f}%")
    print(f"L@150: {single_view_metrics['L@150']:.2f}%")
    print(f"Num: {single_view_metrics['Count']}")

    return overall_metrics, panorama_metrics, single_view_metrics

def cvg_cal_score(similarities, is_panorama, image_locations, text_locations, m=100):
    # val loc is tensor

    total_samples = len(text_locations)
    if isinstance(image_locations, torch.Tensor):
        image_locations = torch.round(image_locations * 1e8) / 1e8
        text_locations = torch.round(text_locations * 1e8) / 1e8
        image_locations = image_locations.tolist()
        text_locations = text_locations.tolist()
    elif isinstance(text_locations, list) and isinstance(text_locations[0], float):
        image_locations = [round(loc, 8) for loc in image_locations]
        text_locations = [round(loc, 8) for loc in text_locations]

    def init_metrics():
        return {"R@1": 0, "R@5": 0, "R@10": 0,
                "L@50": 0, "L@100": 0, "L@150": 0,
                "Deviation": 0, "Count": 0}

    overall_metrics = init_metrics()
    panorama_metrics = init_metrics()
    single_view_metrics = init_metrics()


    top_k_records = []

    print("Calculating R@1, R@5, R@10...")

    for i in range(total_samples):
        original_lat, original_lon = text_locations[i]
        is_pano = is_panorama[i]

        distances = []
        for j, loc in enumerate(image_locations):
            lat, lon = loc
            distance = haversine(original_lat, original_lon, lat, lon)
            distances.append((distance, j))

        distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in distances[:m]]
        sim = similarities[closest_indices, i]
        sorted_indices = np.argsort(-sim)
        best_matches = [image_locations[closest_indices[idx]] for idx in sorted_indices]
        original_image_name = text_locations[i]

        best_match_t = torch.tensor(best_matches[0], dtype=torch.float32)
        original_image_name_t = torch.tensor(original_image_name, dtype=torch.float32)

        best_match_rounded = torch.round(best_match_t * 1e7) / 1e7
        original_image_name_rounded = torch.round(original_image_name_t * 1e7) / 1e7

        label_round = torch.equal(best_match_rounded, original_image_name_rounded)

        if label_round:
            deviation = 0
        else:
            lat_match, lon_match = best_matches[0]
            deviation = haversine(original_lat, original_lon, lat_match, lon_match)

        l_50 = deviation <= 0.05
        l_100 = deviation <= 0.1
        l_150 = deviation <= 0.15

        for metrics in [overall_metrics, panorama_metrics if is_pano else single_view_metrics]:
            metrics["Count"] += 1
            metrics["Deviation"] += deviation
            metrics["L@50"] += int(l_50)
            metrics["L@100"] += int(l_100)
            metrics["L@150"] += int(l_150)

            def to_rounded_tuple(x, scale=1e7):
                if isinstance(x, torch.Tensor):
                    arr = x.detach().cpu().numpy()
                else:
                    arr = np.array(x, dtype=np.float32)
                arr = np.round(arr * scale) / scale
                return tuple(arr.tolist())

            ori_tuple = to_rounded_tuple(original_image_name)
            best_rounded = [to_rounded_tuple(bm) for bm in best_matches]

            metrics["R@1"] += int(ori_tuple == best_rounded[0])
            metrics["R@5"] += int(ori_tuple in best_rounded[:5])
            metrics["R@10"] += int(ori_tuple in best_rounded[:10])

        top_k_records.append({
            'original_image': tuple(original_image_name_rounded.tolist()),
            'top_5_matches': [bm for bm in best_matches[:5]],
            'label': label_round
        })

    def calculate_average(metrics):
        count = metrics["Count"]
        if count == 0:
            return metrics
        return {
            "R@1": (metrics["R@1"] / count) * 100,
            "R@5": (metrics["R@5"] / count) * 100,
            "R@10": (metrics["R@10"] / count) * 100,

            "L@50": (metrics["L@50"] / count) * 100,
            "L@100": (metrics["L@100"] / count) * 100,
            "L@150": (metrics["L@150"] / count) * 100,

            "Deviation": metrics["Deviation"] / count,
            "Count": count
        }

    overall_metrics = calculate_average(overall_metrics)
    panorama_metrics = calculate_average(panorama_metrics)
    single_view_metrics = calculate_average(single_view_metrics)

    print("\nOverall Results:")
    print(f"R@1: {overall_metrics['R@1']:.2f}%")
    print(f"R@5: {overall_metrics['R@5']:.2f}%")
    print(f"R@10: {overall_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {overall_metrics['Deviation']:.2f} km")
    print(f"L@50: {overall_metrics['L@50']:.2f}%")
    print(f"L@100: {overall_metrics['L@100']:.2f}%")
    print(f"L@150: {overall_metrics['L@150']:.2f}%")

    print("\nPanorama Results:")
    print(f"R@1: {panorama_metrics['R@1']:.2f}%")
    print(f"R@5: {panorama_metrics['R@5']:.2f}%")
    print(f"R@10: {panorama_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {panorama_metrics['Deviation']:.2f} km")
    print(f"L@50: {panorama_metrics['L@50']:.2f}%")
    print(f"L@100: {panorama_metrics['L@100']:.2f}%")
    print(f"L@150: {panorama_metrics['L@150']:.2f}%")
    print(f"Num: {panorama_metrics['Count']}")

    print("\nSingle-View Results:")
    print(f"R@1: {single_view_metrics['R@1']:.2f}%")
    print(f"R@5: {single_view_metrics['R@5']:.2f}%")
    print(f"R@10: {single_view_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {single_view_metrics['Deviation']:.2f} km")
    print(f"L@50: {single_view_metrics['L@50']:.2f}%")
    print(f"L@100: {single_view_metrics['L@100']:.2f}%")
    print(f"L@150: {single_view_metrics['L@150']:.2f}%")
    print(f"Num: {single_view_metrics['Count']}")

    return overall_metrics,panorama_metrics,single_view_metrics


def get_pred_batched(qry_emb, pos_emb, all_gallery_ids, qry_batch_size=1000):
    """
    params:
        qry_emb (np.ndarray): shape (N_qry, D).
        pos_emb (np.ndarray): shape (N_pos, D).
        all_gallery_ids (np.ndarray): shape (N_pos,).
        qry_batch_size (int): batched qry.
    """

    all_preds_indices = []
    scores = []
    pos_emb_T = pos_emb.T


    scores_batch = np.dot(qry_emb, pos_emb_T)
    preds_batch_indices = np.argsort(scores_batch, axis=1)[:, ::-1]

    all_preds_indices.append(preds_batch_indices)

    preds_final_indices = np.vstack(all_preds_indices)

    del all_preds_indices
    gc.collect()

    final_preds_ids = all_gallery_ids[preds_final_indices]

    del preds_final_indices
    gc.collect()

    return scores, None, final_preds_ids  # not return scores and preds matrixs，only return the final ID ranking


def get_pred(qry_emb, pos_emb, all_gallery_ids):

    check_matrix(qry_emb, "qry_emb")
    check_matrix(pos_emb, "pos_emb")
    print("Matrix check done.")

    num_qry = qry_emb.shape[0]
    num_pos = pos_emb.shape[0]

    pos_emb_T = pos_emb.T
    scores = np.dot(qry_emb, pos_emb_T)

    preds_indices = np.argsort(scores, axis=1)[:, ::-1]

    final_preds_ids = all_gallery_ids[preds_indices]


    return scores, preds_indices, final_preds_ids


def compute_recall_at_k(cur_query_ids: np.ndarray, preds_id: np.ndarray, ks: tuple = (1, 5, 10)) -> dict:
    recalls = {f'R@{k}': 0.0 for k in ks}
    num_samples = cur_query_ids.shape[0]
    for i in range(num_samples):

        current_query_id = cur_query_ids[i]
        current_gallery_id = preds_id[i]
        for k_val in ks:
            if current_query_id in current_gallery_id[:k_val]:
                recalls[f'R@{k_val}'] += 1

    for k_val in ks:
        recalls[f'R@{k_val}'] = (recalls[f'R@{k_val}'] / num_samples) * 100.0
    return recalls

def set_trainable_layers(model, exclude_modules,lora_target_modules):

    for name, param in model.named_parameters():
        if any(exclude_name in name for exclude_name in exclude_modules):
            param.requires_grad = True


def _process_inputs(batch_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Filters and prepares kwargs for the encoder forward pass."""

    forward_kwargs = {
        k: v for k, v in batch_inputs.items()
        if k in ['input_ids', 'attention_mask', 'pixel_values']
    }

    if 'image_grid_thw' in batch_inputs:
        forward_kwargs['image_grid_thw'] = batch_inputs['image_grid_thw']
    if 'image_sizes' in batch_inputs:
        forward_kwargs['image_sizes'] = batch_inputs['image_sizes']

    if 'input_ids' not in forward_kwargs and 'pixel_values' not in forward_kwargs:
        raise ValueError(
            "Neither 'input_ids' nor 'pixel_values' found. Model requires at least one modality input.")

    return forward_kwargs

@torch.no_grad()
def compute_global_mean_feature(encoder, dataloader, num_batches=10):
    """
    从若干 batch 计算 encoder 输出特征的全局均值。
    """
    encoder.eval()
    qry_feature_sum = 0
    qry_token_count = 0
    pos_feature_sum = 0
    pos_token_count = 0

    for i, batch in enumerate(dataloader):
        my_print(f'batch {i}')
        qry_data, pos_data, building_id, dataset_sample_id = batch
        qry = _process_inputs(qry_data)
        pos = _process_inputs(pos_data)

        qry_outputs = encoder(**qry, output_hidden_states=True, return_dict=True)
        qry_hidden = qry_outputs.hidden_states[-1]

        pos_outputs = encoder(**pos, output_hidden_states=True, return_dict=True)
        pos_hidden = pos_outputs.hidden_states[-1]



        if i >= num_batches:
            break


        qry_feature_sum += qry_hidden.sum(dim=(0, 1))
        qry_token_count += qry_hidden.shape[0] * qry_hidden.shape[1]

        pos_feature_sum += pos_hidden.sum(dim=(0, 1))
        pos_token_count += pos_hidden.shape[0] * pos_hidden.shape[1]

    qry_mean_feature = qry_feature_sum / qry_token_count
    pos_mean_feature = pos_feature_sum / pos_token_count
    return qry_mean_feature.detach().cpu(), pos_mean_feature.detach().cpu()


import torch
import torch.nn as nn
import types




def act_redistribution(model_component, num_registers=4, is_vision=True, is_train= True):
    if hasattr(model_component, "_is_redistributed"):
        return model_component
    model_component._is_redistributed = True
    hidden_size = model_component.config.hidden_size
    reg_name = "v_regs" if is_vision else "t_regs"

    # 1. 初始化寄存器 (检查是否已存在，避免重复定义)
    if not hasattr(model_component, reg_name):
        if not is_train:
            # 【Training-free 模式】：使用全零或正交常数，且不要求梯度
            regs = torch.zeros(1, num_registers, hidden_size)
            # 或者使用特定分布的 Buffer (不会被 load_state_dict 强制要求)
            model_component.register_buffer(reg_name, regs, persistent=False)
        else:
            # 【Trainable 模式】：标准 Parameter，参与反向传播
            regs = nn.Parameter(torch.zeros(1, num_registers, hidden_size))
            nn.init.trunc_normal_(regs, std=0.02)
            setattr(model_component, reg_name, regs)
    orig_forward = model_component.forward
    def forward(self, *args, **kwargs):

        # A. 提取基础特征

        if is_vision:
            # print("is vision")
            pixel_values = args[0] if args else kwargs.get('pixel_values')
            x, _ = self.embeddings(pixel_values)
            print("x.shape:", x.shape)
        else:
            # print("is text")
            x = kwargs.get('inputs_embeds', None)
            if x is None:
                x = self.get_input_embeddings()(kwargs.get('input_ids'))

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # B. 准备寄存器并拼接
        regs = getattr(self, reg_name).expand(batch_size, -1, -1)
        x = torch.cat([regs, x], dim=1)  # 新长度: num_registers + seq_len

        # 1. 处理 Attention Mask
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            m = kwargs["attention_mask"]
            # print("m.shape***********:",m.shape)
            # 为寄存器补全全 1 的 mask
            reg_m = torch.ones((batch_size, num_registers), dtype=m.dtype, device=device)
            new_m = torch.cat([reg_m, m], dim=1)


            kwargs["attention_mask"] = new_m.to(dtype=dtype)

        # 2. 处理 Position IDs
        if not is_vision:
            p = kwargs.get("position_ids", None)
            if p is None:
                p = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            # 寄存器通常占用位置 0
            reg_p = torch.zeros((batch_size, num_registers), dtype=p.dtype, device=device)
            kwargs["position_ids"] = torch.cat([reg_p, p], dim=1)

            if hasattr(self, 'rotary_emb'):
                kwargs["position_embeddings"] = self.rotary_emb(x, kwargs["position_ids"])



        if is_vision:
            outputs  = self.encoder(x,output_attentions=True)
            ori_last_hidden_state , attn_weights = outputs[0], outputs[-1]

        else:
            kwargs.pop('inputs_embeds', None)
            kwargs.pop('position_embeddings', None)
            kwargs.pop('attention_mask', None)
            kwargs.pop('position_ids', None)
            outputs = orig_forward(
                inputs_embeds=x,
                input_ids=None,
                output_attentions=True,
                **kwargs
            )
            ori_last_hidden_state = outputs.last_hidden_state
            attn_weights = outputs.attentions

        # # F. ACT 重分配逻辑 (只在语义 Token 上操作)
        # # p_to_r: [B, H, Seq, num_reg] | p_to_p: [B, H, Seq, Seq]
        # p_to_r_attn = attn_weights[:, :, num_registers:, :num_registers]
        # p_to_p_attn = attn_weights[:, :, num_registers:, num_registers:]
        #
        # sink_weight = p_to_r_attn.sum(dim=-1, keepdim=True)
        # semantic_sum = p_to_p_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        #
        # # 比例瓜分：New_Feature = Old + (Sink * (Old / Total_Semantic))
        # num_heads = attn_weights.shape[1]
        # head_dim = hidden_states.shape[-1] // num_heads
        #
        # # 提取语义 Patch 特征并转换多头形状 [B, H, Seq, Head_Dim]
        # patches_hidden = hidden_states[-1][:, num_registers:, :]
        # patches_heads = patches_hidden.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        #
        # # 显著性筛选
        # mask = ((sink_weight / (sink_weight + semantic_sum)) > incre_ratio).float()
        #
        # # 执行增量分配
        # increment = sink_weight * (patches_heads / semantic_sum)
        # refined_heads = patches_heads + (mask * increment)
        # refined_features = refined_heads.transpose(1, 2).reshape(batch_size, -1, hidden_size)
        #
        # # 合并返回
        # # print("final_layer_outputs:",final_layer_outputs)
        # # print("type(final_layer_outputs):", type(final_layer_outputs))
        # register_hidden =hidden_states[-1][:, :num_registers, :] # [B, num_regs, D]
        # # if hasattr(final_layer_outputs, 'last_hidden_state'):
        #     # 这种方法会保留对象里的 past_key_values, hidden_states 等所有其他字段
        #     # final_layer_outputs.last_hidden_state = refined_features
        # final_layer_outputs.last_hidden_state = hidden_states[-1][:, num_registers:, :]



        # Using register token and EOS token for representation.
        self.last_register_states = ori_last_hidden_state[:, :num_registers, :].detach()
        if is_vision:
            new_attentions = []
            for layer_attn in attn_weights:
                clean_layer_attn = layer_attn[:, :,num_registers:, num_registers:]
                new_attentions.append(clean_layer_attn)
            new_attentions = tuple(new_attentions)

            new_outputs_list = list(outputs)
            new_outputs_list[0] = ori_last_hidden_state[:,num_registers:,:]
            new_outputs_list[-1] = new_attentions


            outputs = tuple(new_outputs_list)

            check_hidden_states = outputs[0]
            check_attentions = outputs[-1]
            return BaseModelOutput(
                last_hidden_state=outputs[0]
            )

        else:

            outputs.last_hidden_state =  ori_last_hidden_state[:,num_registers:,:]
            new_attentions = []
            for layer_attn in attn_weights:
                clean_layer_attn = layer_attn[:, :,num_registers:, num_registers:]
                new_attentions.append(clean_layer_attn)
            new_attentions = tuple(new_attentions)
            outputs.attentions = new_attentions
            new_hiddens = []
            for hidden in outputs.hidden_states:
                clean_hidden = hidden[:, num_registers:, :]
                new_hiddens.append(clean_hidden)
            new_hiddens = tuple(new_hiddens)
            outputs.hidden_states = new_hiddens
            # for key, value in outputs.items():
            #     print(f"key: {key}")
            #     if torch.is_tensor(value):
            #         print(f"shape: {value.shape}")
            #     elif isinstance(value, tuple):
            #         print(f"type: tuple, length: {len(value)}")
            #         print(f"shape for the first: {value[0].shape}")
            #     else:
            #         print(f"type: {type(value)}")
            #     print("-" * 20)

            return outputs

        # # 如果它只是个元组 (Tuple)，我们按索引替换
        # elif isinstance(final_layer_outputs, tuple):
        #     output_list = list(final_layer_outputs)
        #     # output_list[0] = refined_features
        #     output_list[0] = hidden_states[:, num_registers:, :]
        #     output_list.append(hidden_states[:, :num_registers, :])
        #     return tuple(output_list)


    model_component.forward = types.MethodType(forward, model_component)
    return model_component



def apply_attention_redistribution_for_vision(model, signal_index):
    encoder_layers = model.encoder.layer
    num_layers = len(encoder_layers)

    # 1. 统一处理索引逻辑
    if signal_index == ['all']:
        target_indices = list(range(num_layers))
    elif signal_index == [-1]:
        target_indices = [len(encoder_layers) - 1]
    elif isinstance(signal_index, list) and isinstance(signal_index[0], int) :
        target_indices = signal_index
    else:
        raise ValueError("Signal_index must be list.")

    _count = 0
    for i in target_indices:
        if i >= num_layers:
            print(f"!!! Warning: index {i}  out of model layer arrangement !!!")
            continue

        target_attn = encoder_layers[i].attention

        # 2. define package function
        def get_patched_forward(layer_idx):
            # rewrite forward logic by internvl_atten_forward_hook
            def new_forward(self, *args, **kwargs):
                return internvl_vision_atten_forward_hook(self, *args, layer_idx=layer_idx, **kwargs)

            return new_forward

        # 3. 动态替换方法
        target_attn.forward = types.MethodType(get_patched_forward(i), target_attn)
        _count += 1

    print(f"Successfully processed {_count} attention layers (Indices: {target_indices}).")


def apply_attention_redistribution_for_language(model, signal_index):
    encoder_layers = model.layers
    num_layers = len(encoder_layers)

    # 1. 统一处理索引逻辑
    if signal_index == ['all']:
        target_indices = list(range(num_layers))
    elif signal_index == [-1]:
        target_indices = [len(encoder_layers) - 1]
    elif isinstance(signal_index, list) and isinstance(signal_index[0], int) :
        target_indices = signal_index
    else:
        raise ValueError("Signal_index must be list.")

    _count = 0
    for i in target_indices:
        if i >= num_layers:
            print(f"!!! Warning: index {i}  out of model layer arrangement !!!")
            continue

        target_attn = encoder_layers[i].self_attn

        # 2. define package function
        def get_patched_forward(layer_idx):
            # rewrite forward logic by internvl_atten_forward_hook
            def new_forward(self, *args, **kwargs):
                return internvl_language_atten_forward_hook(self, *args, layer_idx=layer_idx, **kwargs)

            return new_forward

        # 3. 动态替换方法
        target_attn.forward = types.MethodType(get_patched_forward(i), target_attn)
        _count += 1

    print(f"Successfully processed {_count} attention layers (Indices: {target_indices}).")

class MMEBModel(pl.LightningModule):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 model_args: ModelArguments,
                 data_args: DataArguments,
                 training_args: TrainingArguments,
                 args_cli,
                 model_backbone: str,
                 save: bool = False,
                 is_train: bool = True
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.config = encoder.config
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.args_cli = args_cli
        self.lr = model_args['learning_rate']
        self.encoder = encoder
        self.pooling = model_args['pooling']
        self.normalize = model_args['normalize']
        self.temperature = model_args['temperature']
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        # if self.args_cli.fea_dim>0:
        #     self.linear_layer = nn.Linear(encoder.config.text_config.hidden_size, self.args_cli.fea_dim)
        self.model_backbone = model_backbone

        self.save = save
        self.has_visualized = False
        self.visualize_dir = "gpfs2/scratch/ychen57/code/LightVec/output/vis/"

        self.is_train = is_train
        #input_dim: int, num_heads: int, num_queries: int = 1, num_layers
        # if args_cli.fea_token == 'query':
        #     self.qry_pooling_module = TransformerDecoderPooling(encoder.config.vision_config.hidden_size, num_queries = self.args_cli.query, \
        #                                                         num_heads = self.args_cli.head_num, num_layers = self.args_cli.layer)
        #     self.pos_pooling_module = TransformerDecoderPooling(encoder.config.vision_config.hidden_size, num_queries = self.args_cli.query, \
        #                                                         num_heads = self.args_cli.head_num, num_layers = self.args_cli.layer)



        if self.save:
            slurm_job_name = args_cli.job_name_from_slurm
            if args_cli.checkpoint_to_eval is not None:
                save_log_name = str(slurm_job_name) + str(
                    os.path.basename(args_cli.checkpoint_to_eval).replace('.ckpt', ''))
            else:
                save_log_name = str(slurm_job_name) + "_origin"


            if self.args_cli.checkpoint_to_eval is not None:
                print("checkpoint_to_eval", self.args_cli.checkpoint_to_eval)
                print("os.path.basename(self.args_cli.checkpoint_to_eval):",os.path.basename(self.args_cli.checkpoint_to_eval))
                self.eval_output_base_dir = os.path.dirname(self.args_cli.checkpoint_to_eval)
                if 'Epoch' in self.args_cli.checkpoint_to_eval:
                    epoch = self.args_cli.checkpoint_to_eval.replace('.ckpt', '').split('/')[-1].split('Epoch')[-1]
                elif 'last' in self.args_cli.checkpoint_to_eval:
                    epoch = 'last'
                else:
                    assert 'cannot find epoch' in self.args_cli.checkpoint_to_eval

                self.features_cache_dir = os.path.join(self.eval_output_base_dir,
                                                       f"cached_features")
                os.makedirs(self.features_cache_dir, exist_ok=True)
                self.qry_embeddings_path = os.path.join(self.features_cache_dir,
                                                        f"qry_embeddings_Epoch" + epoch + ".npy")
                self.pos_embeddings_path = os.path.join(self.features_cache_dir,
                                                        f"pos_embeddings_Epoch" + epoch + ".npy")
                self.ids_path = os.path.join(self.features_cache_dir,
                                             f"ids_Epoch" + epoch + ".npy")
                if "text2" in self.args_cli.task:
                    self.qry_texts_path = os.path.join(self.features_cache_dir, f"{save_log_name}_qry_texts_{self.data_args['subset_name']}.npy")
                    self.pos_images_path = os.path.join(self.features_cache_dir,
                                                       f"{save_log_name}_pos_images_{self.data_args['subset_name']}.npy")


            else:
                self.eval_output_base_dir = self.args_cli.eval_output_dir
                self.output_filename_stem = self.eval_output_base_dir + "/"
                self.features_cache_dir = os.path.join(self.eval_output_base_dir,
                                                       f"cached_features")

                os.makedirs(self.features_cache_dir, exist_ok=True)

                self.qry_embeddings_path = os.path.join(self.features_cache_dir,
                                                        f"{save_log_name}_qry_embeddings_{self.data_args['subset_name']}.npy")
                self.pos_embeddings_path = os.path.join(self.features_cache_dir,
                                                        f"{save_log_name}_pos_embeddings_{self.data_args['subset_name']}.npy")
                self.ids_path = os.path.join(self.features_cache_dir,
                                             f"{save_log_name}_ids_{self.data_args['subset_name']}.npy")

                if  "text2" in self.args_cli.task:
                    self.qry_texts_path = os.path.join(self.features_cache_dir, f"{save_log_name}_qry_texts_{self.data_args['subset_name']}.npy")
                    self.pos_images_path = os.path.join(self.features_cache_dir,
                                                       f"{save_log_name}_pos_images_{self.data_args['subset_name']}.npy")
            my_print("self.ids_path：", self.ids_path)
            my_print("self.qry_embeddings_path：", self.qry_embeddings_path)
            my_print("self.pos_embeddings_path：", self.pos_embeddings_path)

    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments, args_cli,save, is_train,train_dataloader=None,**kwargs):
        config = AutoConfig.from_pretrained(model_args['model_name_or_path'], trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        my_print("************ model_backbone **************: ", model_backbone)
        hf_model_type = config.model_type

        if hf_model_type not in backbone2model:
            raise ValueError(f"Model type '{hf_model_type}' from config.json is not supported in MODEL2BACKBONE.")

        # Loading the base model
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        if model_backbone in [QWEN2_VL, QWEN2_5_VL, PHI3V]:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args['model_name_or_path'],
                config=config,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            base_model.gradient_checkpointing_enable()
        elif model_backbone == INTERNVL:
            my_print(f"Loading InternVL; Base model: {model_args['model_name_or_path']}")
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args['model_name_or_path'],
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if hasattr(base_model.vision_tower, "config"):
                base_model.vision_tower.config.output_attentions = True
            # fea_token:[redis_eos,eos, redis_two, redis_all]
            if args_cli.num_registers > 0:
                reg_name = "t_regs"
                if not is_train:
                    # 【Training-free 模式】：使用全零或正交常数，且不要求梯度
                    regs = torch.ones(1, args_cli.num_registers, base_model.language_model.config.hidden_size)
                    torch.nn.init.trunc_normal_(regs, std=0.02)  # 0.02 是常用的初始化标准差
                    base_model.language_model.layers.register_buffer(reg_name, regs, persistent=False)
                else:
                    # 【Trainable 模式】：标准 Parameter，参与反向传播
                    if args_cli.init == "zeros":
                        regs = nn.Parameter(torch.zeros(1, args_cli.num_registers, base_model.language_model.config.hidden_size))
                    elif args_cli.init == "ones":
                        regs = nn.Parameter(torch.ones(1, base_model.language_model.config.hidden_size))
                    nn.init.trunc_normal_(regs, std=0.02)
                    setattr(base_model.language_model.layers, reg_name, regs)
            if args_cli.num_registers and args_cli.fea_token == 'redis_eos':
                base_model.language_model = act_language_register_without_position(base_model.language_model,
                                                                                   num_registers=args_cli.num_registers,
                                                                                   is_vision=False, is_train=is_train)

            # if args_cli.num_registers == 0 and (args_cli.fea_token == 'redis_eos' or args_cli.fea_token == 'redis_two'):
            #     # base_model.language_model = act_redistribution(base_model.language_model, num_registers=args_cli.num_registers,is_vision=False, is_train=is_train)
                # apply_attention_redistribution_for_language(base_model.language_model, signal_index=[-1])
                # apply_attention_redistribution_for_vision(base_model.vision_tower, signal_index=[-1])
            # elif args_cli.num_registers == 0 and args_cli.fea_token == 'redis_all':
            #     apply_attention_redistribution_for_language(base_model.language_model, signal_index=['all'])
            #     # apply_attention_redistribution_for_vision(base_model.vision_tower, signal_index=['all'])

        elif model_backbone == IDEFICS3 or model_backbone == SMOLVLM:  # Now handling it as IDEFICS3
            my_print(f"Loading Idefics3; Base model: {model_args['model_name_or_path']}")
            from src.vlm_backbone.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
            from src.vlm_backbone.smolvlm.configuration_smolvlm import SmolVLMConfig

            if '500' in model_args['model_name_or_path']:
                local_model_path = '/gpfs2/scratch/ychen57/code/LightVec/src/vlm_backbone/smolvlm500/'
            elif '256' in model_args['model_name_or_path']:
                local_model_path = '/gpfs2/scratch/ychen57/code/LightVec/src/vlm_backbone/smolvlm/'
            else:
                assert "not matched config document for smolvlm"

            config = SmolVLMConfig.from_pretrained(local_model_path)

            base_model = SmolVLMForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=local_model_path,
                                                                         attn_implementation="eager",
                                                                         torch_dtype=torch.bfloat16,
                                                                         config=config)

        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args['model_name_or_path'], **kwargs, config=config,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        print("model_args['lora']:",model_args['lora'])
        if model_args['lora']:
            target_modules = model_args['lora_target_modules'].split(',')
            print("target_modules:", target_modules)
            print("model_args['lora_alpha']:",model_args['lora_alpha'])
            print("lora_r:",model_args['lora_r'])
            if model_backbone in [QWEN2_VL, QWEN2_5_VL, PHI3V]:
                use_dora = False
            else:
                use_dora = True
            if is_train:
                lora_config = LoraConfig(
                    r=model_args['lora_r'],
                    lora_alpha=model_args['lora_alpha'],
                    target_modules=target_modules,
                    modules_to_save=["v_regs", "t_regs"],
                    lora_dropout=model_args['lora_dropout'],
                    init_lora_weights="gaussian",
                    use_dora=use_dora,
                    inference_mode=not is_train,
                )
            else:
                lora_config = LoraConfig(
                    r=model_args['lora_r'],
                    lora_alpha=model_args['lora_alpha'],
                    target_modules=target_modules,
                    lora_dropout=model_args['lora_dropout'],
                    init_lora_weights="gaussian",
                    use_dora=use_dora,
                    inference_mode=not is_train,
                )

            # Apply LoRA first to the entire model
            lora_model = get_peft_model(base_model, lora_config)
            if is_train:
                for name, param in lora_model.named_parameters():
                    if any(reg in name for reg in ["v_regs", "t_regs"]):
                        param.requires_grad = True

            model = cls(
                encoder=lora_model,
                model_backbone=model_backbone,
                model_args=model_args,
                training_args=training_args,
                data_args=data_args,
                args_cli=args_cli,
                save=save,
                is_train = is_train,
            )
        else:
            model = cls(
                encoder=base_model,
                model_args=model_args,
                training_args=training_args,
                data_args=data_args,
                args_cli=args_cli,
                model_backbone=model_backbone,
                save=save,
                is_train=is_train,
                # init_qry_mean_feature=init_qry_mean_feature,
                # init_pos_mean_feature=init_pos_mean_feature,
            )
        # 检查模型中第一个线性层的类型
        for name, module in base_model.named_modules():
            if "proj" in name or "fc" in name:  # 寻找线性层
                print(f"Layer: {name}")
                print(f"Module Type: {type(module)}")
                # 如果使用了 4-bit 量化，这里通常会显示 <class 'bitsandbytes.nn.modules.Linear4bit'>

                if hasattr(module, 'weight'):
                    print(f"Weight dtype: {module.weight.dtype}")
                    # 注意：对于 4-bit 模型，weight.dtype 可能会显示为 torch.uint8
                    # 因为 4位的数据是打包存储在 uint8 里的
                break

        def memory_stats():
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        for name, param in base_model.named_parameters():
            if any(reg in name for reg in ["v_regs", "t_regs", "registers"]):
                print(f">>> Register Component Found: {name} | Trainable: {param.requires_grad} | Shape: {param.shape}")

        memory_stats()
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def _pooling(self, hidden_state, attention_mask,attn=None,reg_state=None):
        if 'eos' in self.args_cli.fea_token:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = hidden_state[torch.arange(batch_size), -1, :]

            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = hidden_state[
                        torch.arange(batch_size, device=hidden_state.device), eos_indices,:]
        elif 'all' in self.args_cli.fea_token:
            batch_size = hidden_state.shape[0]
            all_reps = hidden_state[
                   torch.arange(batch_size, device=hidden_state.device), :, :]
            reps = all_reps.mean(dim=1)

        elif self.args_cli.fea_token == 'redis_two':
            batch_size = hidden_state.shape[0]
            device = hidden_state.device
            num_regs = self.args_cli.num_registers

            # 1. 提取 EOS 特征 [batch_size, hidden_size]
            # 使用高级索引定位每个样本的最后一个有效 token
            eos_indices = attention_mask.sum(dim=1) - 1
            eos_reps = hidden_state[torch.arange(batch_size, device=device), eos_indices, :]

            # 2. 提取寄存器特征 [batch_size, num_registers, hidden_size]
            # [batch_size, num_registers, hidden_size]
            reg_reps = reg_state


            # 方案 B: 如果你只想把它们压平做最后的表示 (例如 [B, (num_regs+1)*D])
            reps = torch.cat([reg_reps.flatten(1), eos_reps], dim=1)
        return reps




    def _process_inputs(self,batch_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Filters and prepares kwargs for the encoder forward pass."""

        forward_kwargs = {
            k: v for k, v in batch_inputs.items()
            if k in ['input_ids', 'attention_mask', 'pixel_values']
        }

        if 'image_grid_thw' in batch_inputs:
            forward_kwargs['image_grid_thw'] = batch_inputs['image_grid_thw']
        if 'image_sizes' in batch_inputs:
            forward_kwargs['image_sizes'] = batch_inputs['image_sizes']

        if 'input_ids' not in forward_kwargs and 'pixel_values' not in forward_kwargs:
            raise ValueError(
                "Neither 'input_ids' nor 'pixel_values' found. Model requires at least one modality input.")

        if 'attention_mask' not in forward_kwargs and self.pooling in ['last', 'eos']:
            if 'input_ids' in forward_kwargs and forward_kwargs['input_ids'] is not None:
                forward_kwargs['attention_mask'] = torch.ones_like(forward_kwargs['input_ids'])
            else:
                print(
                    "Warning: EOS/Last pooling selected but no attention mask or input_ids provided. Results may be suboptimal.")
        return forward_kwargs

    def forward(self, qry_inputs: Dict[str, Tensor],  pos_inputs: Dict[str, Tensor]) -> Tensor:
        """
        Main forward pass for the model. It takes a batch_inputs dictionary
        and returns the pooled representations.
        """
        qry_kwargs = self._process_inputs(qry_inputs)
        pos_kwargs = self._process_inputs(pos_inputs)
        qry_outputs = self.encoder(**qry_kwargs, output_hidden_states=True, output_attentions=True,return_dict=True)
        qry_att = qry_kwargs.get('attention_mask')
        pos_outputs = self.encoder(**pos_kwargs, output_hidden_states=True,output_attentions=True, return_dict=True)

        pos_att = pos_kwargs.get('attention_mask')
        # if self.args_cli.fea_token == 'query':
        #     qry_last_hidden_state = qry_outputs.hidden_states[-1]
        #     pos_last_hidden_state = pos_outputs.hidden_states[-1]
        #     qry_agg, pos_agg = self.cross_attention_aggregation_pooling(qry_last_hidden_state=qry_last_hidden_state, qry_attention_mask=qry_att, \
        #                                      pos_last_hidden_state=pos_last_hidden_state, pos_attention_mask=pos_att)
        #
        #     qry_fea = F.normalize(qry_agg, p=2, dim=-1)
        #     pos_fea = F.normalize(pos_agg, p=2, dim=-1)
        #
        # elif self.args_cli.fea_token == 'eos' or self.args_cli.fea_token == 'whole':
        extra_outputs = {}

        # 提取 language model 的状态
        # if hasattr(self.encoder.language_model, 'last_register_states'):
        #     extra_outputs['text_register_states'] = self.encoder.language_model.last_register_states
        #
        # # 提取 vision model 的状态
        # if hasattr(self.encoder.vision_tower, 'last_register_states'):
        #     extra_outputs['vision_register_states'] = self.encoder.vision_tower.last_register_states

        qry_attn = qry_outputs.attentions[-1][0]
        pos_attn = pos_outputs.attentions[-1][0]


        qry_last_hidden_state = qry_outputs.hidden_states[-1]
        pos_last_hidden_state = pos_outputs.hidden_states[-1]

        # fea_token: [redis_eos, eos, redis_two, redis_all]
        if 'eos' in self.args_cli.fea_token or 'all' in self.args_cli.fea_token:
            qry_fea = self._pooling(hidden_state=qry_last_hidden_state, attention_mask=qry_att, attn=None,reg_state=None)
            pos_fea = self._pooling(hidden_state=pos_last_hidden_state, attention_mask=pos_att, attn=None,reg_state=None)
        elif self.args_cli.fea_token == 'redis_two':
            qry_fea =self._pooling(hidden_state=qry_last_hidden_state, attention_mask=qry_att,attn = qry_attn, reg_state = extra_outputs['text_register_states'])
            pos_fea =self._pooling(hidden_state=pos_last_hidden_state, attention_mask=pos_att,attn = pos_attn, reg_state = extra_outputs['vision_register_states'])

        # if self.args_cli.fea_dim>0:
        #     qry_fea = self.linear_layer(qry_fea)
        #     pos_fea = self.linear_layer(pos_fea)

        qry_fea = torch.nn.functional.normalize(qry_fea, p=2, dim=-1)
        pos_fea = torch.nn.functional.normalize(pos_fea, p=2, dim=-1)

        return qry_fea, pos_fea



    def training_step(self, batch, batch_idx,target: Tensor = None):
        qry_data, pos_data, building_id,dataset_sample_id = batch

        qry_reps, pos_reps = self(qry_inputs = qry_data,pos_inputs = pos_data)
        scores = self.compute_similarity(qry_reps, pos_reps)
        logits = scores / self.temperature
        if target is None:
            target_per_qry = pos_reps.size(0) // qry_reps.size(0)
            target = torch.arange(
                0, qry_reps.size(0) * target_per_qry, target_per_qry, device=qry_reps.device, dtype=torch.long)
        loss= self.cross_entropy(logits, target)
        torch.cuda.empty_cache()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.args_cli.batch_size)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss



    def validation_step(self, batch, batch_idx):
        if "GeoText" in self.data_args['json_path']:
            qry_data, pos_data, ids, qry_texts, raw_images  = batch
        elif "CVGText" in self.data_args['json_path']:
            qry_data, pos_data, ids, _,_ = batch

        qry_reps,pos_reps = self(qry_data,pos_data)
        print("qry_reps:", qry_reps.shape)
        scores = self.compute_similarity(qry_reps, pos_reps)
        logits = scores / self.temperature

        target_per_qry = pos_reps.size(0) // qry_reps.size(0)
        target = torch.arange(
            0, qry_reps.size(0) * target_per_qry, target_per_qry,
            device=qry_reps.device, dtype=torch.long
        )

        val_loss = self.cross_entropy(logits, target)
        self.log(
            'val_loss', val_loss,
            on_step=True, on_epoch=True,
            prog_bar=False, logger=True, sync_dist=True
        )
        if self.data_args['dataset_name'] == 'GeoText':
            ids = np.array(ids, dtype=np.int64)


        self.val_qry_embeddings.append(qry_reps.cpu().float().numpy())
        self.val_pos_embeddings.append(pos_reps.cpu().float().numpy())
        self.val_ids.append(ids)

        return {}

    def is_panorama(self, filename):
            """
            Determine if the file is a panoramic image.
            Panoramic filenames generally have more metadata segments compared to single-view images.
            """
            parts = filename.split("_")
            return len(parts) > 2

    def on_validation_epoch_start(self):
        self.val_qry_embeddings = []
        self.val_pos_embeddings = []
        self.val_ids = []

    def on_validation_epoch_end(self):
        res = {"R@1": 0, "R@5": 0, "R@10": 0,
         "L@50": 0, "L@100": 0, "L@150": 0,
         "Deviation": 0, "Count": 0}

        if self.data_args['dataset_name'] == "GeoText":
            local_qry_emb = torch.from_numpy(np.concatenate(self.val_qry_embeddings, axis=0)).to(self.device)
            local_pos_emb = torch.from_numpy(np.concatenate(self.val_pos_embeddings, axis=0)).to(self.device)


            all_qry_tensor_np = local_qry_emb.cpu().numpy()
            all_pos_tensor_np = local_pos_emb.cpu().numpy()
            all_ids_np = np.concatenate(self.val_ids, axis=0)

            scores_qry2pos, _, preds_id_qry2pos = get_pred_batched(
                all_qry_tensor_np, all_pos_tensor_np, all_ids_np
            )
            res_qry2pos = compute_recall_at_k( all_ids_np, preds_id_qry2pos)

            res = res_qry2pos
            my_print("res:", res)


        elif self.data_args['dataset_name'] == "CVGText":
            local_qry_emb = torch.from_numpy(np.concatenate(self.val_qry_embeddings, axis=0)).to(self.device)
            local_pos_emb = torch.from_numpy(np.concatenate(self.val_pos_embeddings, axis=0)).to(self.device)

            all_qry_tensor_np = local_qry_emb.cpu().numpy()
            all_pos_tensor_np = local_pos_emb.cpu().numpy()
            if isinstance(self.val_ids[0], (list, np.ndarray)):
                all_ids = [x for sublist in self.val_ids for x in sublist]
            else:
                all_ids = [x for x in self.val_ids]
            is_pano = [self.is_panorama(id) for id in all_ids]

            all_ids = [extract_lat_lon(x) for x in all_ids]

            image_filenames_list = all_ids
            text_filenames_list = all_ids
            sim_qry2pos = np.dot(all_qry_tensor_np, all_pos_tensor_np.T)
            res,panorama_metrics, single_view_metrics = cvg_cal_score(similarities=sim_qry2pos, is_panorama=is_pano, \
                                image_locations=image_filenames_list, text_locations=text_filenames_list, m=100)

            print("results:", res)

        self.log("val/recall@1", res["R@1"],on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/recall@5", res["R@5"],on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/recall@10", res["R@10"],on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def configure_optimizers(self):
        # optimizer: using linear strategy and lr will decreasing util it became 5e-8 (yuki on oct. 6)
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.model_args['weight_decay'],
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_args['learning_rate'])
        for name, param in self.named_parameters():
            if any(reg in name for reg in ["v_regs", "t_regs", "registers"]):
                print(f">>> Register Component Found: {name} | Trainable: {param.requires_grad} | Shape: {param.shape}")

        # 2. Computing total steps for training
        total_steps = self.training_args['steps']

        # 3.Enhanced optimizer —— the lowest lr is not 0, but a low number (ie. 1e-6)
        # lower lr ratio, which is the final lr = initialized lr * * min_lr_ratio
        min_lr_ratio = 1e-3
        min_lr = self.model_args['learning_rate'] * min_lr_ratio

        def lr_lambda(current_step):
            # linear warmup
            if current_step < self.training_args['warmup_steps']:
                return float(current_step) / float(max(1, self.training_args['warmup_steps']))
            # Linear decay is applied until the minimum learning rate (min_lr) is reached
            # Progress is the ratio used to define this decay schedule.
            progress = (current_step - self.training_args['warmup_steps']) / float(
                max(1, total_steps - self.training_args['warmup_steps']))

            # 1.0 - progress * (1.0 - min_lr_ratio): when finishing warmup, it should be 1.0
            # ‘progress * (1.0 - min_lr_ratio): This term indicates how much decay has been covered from the end of the warmup phase to the current point.
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

        # the second para lr_lambda refer a function，which is to calculate the multiplier of lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def test_step(self, batch: tuple, batch_idx: int):
        qry, pos, cur_id, qry_texts, pos_images= batch
        qry_reps, pos_reps = self(qry, pos)
        self.all_qry_embeddings.append(qry_reps.cpu().float().numpy())
        self.all_pos_embeddings.append(pos_reps.cpu().float().numpy())
        self.all_ids.append(cur_id)
        if self.data_args['dataset_name'] == "GeoText":
            self.qry_texts.append(qry_texts)
            self.pos_images.append(pos_images)
        return {}


    def on_test_epoch_start(self):
        self.all_qry_embeddings = []
        self.all_pos_embeddings = []
        self.all_ids = []

        if self.data_args['dataset_name'] == "GeoText":
            self.qry_texts = []
            self.pos_images = []




    def on_test_epoch_end(self):
        if self.data_args['dataset_name'] == "GeoText":
            if self.save:
                if not (os.path.exists(self.qry_embeddings_path) and os.path.exists(self.pos_embeddings_path) and os.path.exists(self.ids_path)):
                    np.save(file=self.qry_embeddings_path, arr=np.concatenate(self.all_qry_embeddings, axis=0))
                    np.save(file=self.pos_embeddings_path, arr=np.concatenate(self.all_pos_embeddings, axis=0))
                    np.save(file=self.ids_path, arr=np.concatenate(self.all_ids, axis=0))
                    if self.args_cli.task =="text2image":
                        np.save(file=self.qry_texts_path, arr=np.concatenate(self.qry_texts, axis=0))
                        np.save(file=self.pos_images_path, arr=np.concatenate(self.pos_images, axis=0))


            local_qry_emb = torch.from_numpy(np.concatenate(self.all_qry_embeddings, axis=0)).to(self.device)
            local_pos_emb = torch.from_numpy(np.concatenate(self.all_pos_embeddings, axis=0)).to(self.device)
            fea_dim = local_qry_emb.shape[-1]
            first_id_element = None
            if self.all_ids and len(self.all_ids) > 0:
                if isinstance(self.all_ids[0], (list, np.ndarray)):
                    if len(self.all_ids[0]) > 0:
                        first_id_element = self.all_ids[0][0]
                else:
                    first_id_element = self.all_ids[0]

            all_qry_tensor_np = local_qry_emb.cpu().numpy()
            all_pos_tensor_np = local_pos_emb.cpu().numpy()
            all_ids_np = np.concatenate(self.all_ids, axis=0)
            if self.args_cli.task == "text2image":
                print("start compute predict")
                scores_qry2pos, _, preds_id_qry2pos = get_pred_batched(
                    all_qry_tensor_np, all_pos_tensor_np, all_ids_np
                )
                print("start compute recall")
                res = compute_recall_at_k( all_ids_np, preds_id_qry2pos)

                my_print("result for text2image:", res)

            elif self.args_cli.task == "image2text":
                scores_pos2qry, _, preds_id_pos2qry = get_pred(
                 all_pos_tensor_np, all_qry_tensor_np, all_ids_np
                )
                res = compute_recall_at_k(all_ids_np, preds_id_pos2qry)

                my_print("result for image2text:", res)

            checkpoint_name = self.args_cli.checkpoint_to_eval.replace('.ckpt', '')
            json_output_path = os.path.join(checkpoint_name + str(self.args_cli.task)+"_res.json")
            # Get the directory name from the file path
            output_dir = os.path.dirname(json_output_path)

            # Create the directory if it doesn't exist.
            # The `exist_ok=True` argument prevents an error if the directory already exists.
            os.makedirs(output_dir, exist_ok=True)

            with open(json_output_path, 'w') as f:
                json.dump({"res_"+str(self.args_cli.task): res}, f, indent=4)
            my_print("result save to:", json_output_path)

        elif self.data_args['dataset_name'] == "CVGText":
            if self.save:
                if not (os.path.exists(self.qry_embeddings_path) and os.path.exists(self.pos_embeddings_path) and os.path.exists(self.ids_path)):
                    np.save(file=self.qry_embeddings_path, arr=np.concatenate(self.all_qry_embeddings, axis=0))
                    np.save(file=self.pos_embeddings_path, arr=np.concatenate(self.all_pos_embeddings, axis=0))
                    np.save(file=self.ids_path, arr=np.concatenate(self.all_ids, axis=0))
            local_qry_emb = torch.from_numpy(np.concatenate(self.all_qry_embeddings, axis=0)).to(self.device)
            local_pos_emb = torch.from_numpy(np.concatenate(self.all_pos_embeddings, axis=0)).to(self.device)


            if isinstance(self.all_ids[0], (list, np.ndarray)):
                local_ids = [str(x) for sublist in self.all_ids for x in sublist]
            else:
                local_ids = [str(x) for x in self.all_ids]
            is_pano = [self.is_panorama(id) for id in local_ids]
            all_qry_tensor_np = local_qry_emb.cpu().numpy()
            all_pos_tensor_np = local_pos_emb.cpu().numpy()
            if isinstance(self.all_ids[0], (list, np.ndarray)):
                all_ids = [str(x) for sublist in self.all_ids for x in sublist]
            else:
                all_ids = [str(x) for x in self.all_ids]
            # image_file_name = all_ids
            all_ids = [extract_lat_lon(id) for id in all_ids]
            image_filenames_list = all_ids
            text_filenames_list = all_ids
            sim_qry2pos = np.dot(all_qry_tensor_np, all_pos_tensor_np.T)

                # top5_indices = np.argsort(-sim_qry2pos, axis=1)[:, :5]  # descending order
                # top1_indices = top5_indices[:, 0]
                #
                # r5_but_not_r1 = []
                # if self.args_cli.task!="cross_area":
                #     for i in range(sim_qry2pos.shape[0]):
                #         correct_index = i  # ground truth (query i and the corresponding pos i)
                #         ## Save R@5 correct but R@1 fail example
                #         if correct_index in top5_indices[i] and correct_index != top1_indices[i]:
                #         ## save R@1 CORRECT example
                #         # if correct_index == top1_indices[i]:
                #             # gathering information
                #             r5_but_not_r1.append({
                #                 "query_id": image_file_name[i],
                #                 "top1_pred": image_file_name[top1_indices[i]],
                #                 "top5_preds": [image_file_name[j] for j in top5_indices[i].tolist()],
                #                 "similarities_top5": sim_qry2pos[i, top5_indices[i]].tolist()
                #             })
                #
                #             sample = r5_but_not_r1[-1]
                #             query_id = sample["query_id"]
                #             top1_path = sample["top1_pred"]
                #             top5_list = sample["top5_preds"]
                #             print("*********** i **********:",i)
                #             # making children file for each query.
                #             save_root = \
                #             self.args_cli.checkpoint_to_eval.split(self.args_cli.checkpoint_to_eval.split('/')[-1])[0]
                #             query_folder = os.path.join(save_root, f"sample_{i:05d}")
                #             os.makedirs(query_folder, exist_ok=True)
                #
                #             # copy the picture of query itself
                #             if os.path.exists(os.path.join(self.data_args['image_dir'],query_id)):
                #                 shutil.copy(os.path.join(self.data_args['image_dir'],query_id), os.path.join(query_folder, "query"+str(query_id.split('/')[-1])))
                #             else:
                #                 print(f"Warning: query image not found: {query_id}")
                #
                #             # copy top1
                #             if os.path.exists(os.path.join(self.data_args['image_dir'],top1_path)):
                #                 shutil.copy(os.path.join(self.data_args['image_dir'],top1_path), os.path.join(query_folder, "top1.jpg"))
                #             else:
                #                 print(f"Warning: top1 image not found: {top1_path}")
                #
                #             # copy top5
                #             for rank, path in enumerate(top5_list):
                #                 if os.path.exists(os.path.join(self.data_args['image_dir'],path)):
                #                     shutil.copy(os.path.join(self.data_args['image_dir'],path), os.path.join(query_folder, f"top5-top{rank + 1}.jpg"))
                #                 else:
                #                     print(f"Warning: top{rank + 1} image not found: {path}")
                #
                #             print(f"✅all samples are save at: {query_folder}/")
                #         if len(r5_but_not_r1) > 10:
                #             break
                #
                #     # save as JSON
                #     checkpoint_name = self.args_cli.checkpoint_to_eval.replace('.ckpt', '')
                #     r5_not_r1_path = os.path.join(checkpoint_name + "_fail.json")
                #
                #     with open(r5_not_r1_path, 'w') as f:
                #         json.dump(r5_but_not_r1, f, indent=4)
                #
                #     my_print(f"Saved {len(r5_but_not_r1)} samples where R@5 correct but R@1 wrong to {r5_not_r1_path}")

                # top5_indices = np.argsort(-sim_qry2pos, axis=1)[:, :5]  # descending order
                # top1_indices = top5_indices[:, 0]

            res_t2i, panorama_res_t2i, single_view_res_t2i = cvg_cal_score_unified(similarities=sim_qry2pos,
                                                                                       is_panorama=is_pano, \
                                                                                       image_locations=image_filenames_list,
                                                                                       text_locations=text_filenames_list,
                                                                                       m=100, mode='text2image')
            res_i2t, panorama_res_i2t, single_view_res_i2t = cvg_cal_score_unified(similarities=sim_qry2pos,
                                                                                   is_panorama=is_pano, \
                                                                                   image_locations=image_filenames_list,
                                                                                   text_locations=text_filenames_list,
                                                                                   m=100, mode='image2text')



            my_print("results for text2image:", res_t2i)
            my_print("results for image2text:", res_i2t)

            checkpoint_name = self.args_cli.checkpoint_to_eval.replace('.ckpt', '')
            if 'cross' in self.args_cli.task:
                json_output_path = os.path.join(checkpoint_name + '-to-' + self.args_cli.target + "_result.json")
            else:
                json_output_path = os.path.join(checkpoint_name + "_result.json")
            # if self.args_cli.task != "cross_area":
            with open(json_output_path, 'w') as f:
                json.dump({
                    "res_t2i": res_t2i,
                    "panorama_res_t2i": panorama_res_t2i,
                    "single_view_res_t2i": single_view_res_t2i,
                    "res_i2t": res_i2t,
                    "panorama_res_i2t": panorama_res_i2t,
                    "single_view_res_i2t": single_view_res_i2t
                }, f, indent=4)
                my_print("result save to:", json_output_path)






