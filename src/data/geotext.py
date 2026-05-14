import datasets
import os
import json
import numpy as np
from src.model_utils import PHI3V, vlm_image_tokens, QWEN2_5_VL,QWEN2_VL,SMOLVLM
from typing import List, Union
import functools
from src.model_utils import process_vlm_inputs_fns

from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
from dataclasses import dataclass
from PIL import Image
import torch
from collections import defaultdict
from pytorch_lightning.utilities import rank_zero_only
from src.data.augmentation import geotext_train_transform,random_transform,\
    geometry_train_transform,puregeo_train_transform,\
geometry_without_Elastic_train_transform,geometry_without_Color_train_transform,\
geometry_without_ColorAndElastic_train_transform
import re

@rank_zero_only
def my_print(*args, **kwargs):
    print(*args, **kwargs)


def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    # if resolution == "high":
    #     image = image.resize((1344, 1344))
    # elif resolution == "mid":
    #     image = image.resize((672, 672))
    # elif resolution == "low":
    #     image = image.resize((128, 128))
    if type(resolution) == int:
        image = image.resize((resolution, resolution))
    else:
        cur_max_dim = max(image.size)

        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

@dataclass
class TrainTextImageDataCollator:
    data_args: object
    model_args: object
    cli_args: object
    processor: object
    model_backbone_name: str
    is_train: bool

    def _stack_pixel_values(self, pixel_values_list, reference_input_ids_for_batch_size):
        if not any(pv is not None for pv in pixel_values_list):
            return None
        return torch.stack(pixel_values_list, dim=0)

    def _ensure_tensor_and_type(self, data, dtype):
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        try:
            return torch.tensor(data, dtype=dtype)
        except Exception:
            if isinstance(data, list) and all(isinstance(x, list) for x in data):
                return torch.stack([torch.tensor(x, dtype=dtype) for x in data])
            raise

    def __call__(self, examples):
        qry_texts, qry_pil_images, pos_texts, pos_pil_images, ids, sample_ids = [], [], [], [], [], []

        for ex, sample_id in examples:
            qry_text = ex.get("qry_text")
            qry_img_data = ex.get("qry_img_path")
            pos_text = ex.get("pos_text")
            pos_img_data = ex.get("pos_img_path")

            building_id = ex.get("building_id")
            ids.append(building_id)
            sample_ids.append(sample_id)

            if self.cli_args.augmentation == 'random':
                transform = random_transform()
            elif self.cli_args.augmentation == 'geotext':
                transform = geotext_train_transform()
            elif self.cli_args.augmentation == 'geometry_without_Elastic':
                transform = geometry_without_Elastic_train_transform()
            elif self.cli_args.augmentation == 'None' or self.cli_args.augmentation =='none':
                transform = None

            if isinstance(qry_img_data, str) and qry_img_data != '-':
                if self.is_train and transform is not None:
                    qry_img_data = transform(qry_img_data)
                width, height = qry_img_data.size
                print(f"Image width: {width}, Image height: {height}")

            qry_pil_images.append(qry_img_data)

            if isinstance(pos_img_data, str) and pos_img_data != '-':
                if self.is_train and transform is not None:
                    pos_img_data = transform(pos_img_data)
                width, height = pos_img_data.size
                print(f"Image width: {width}, Image height: {height}")

            pos_pil_images.append(pos_img_data)

            qry_texts.append(qry_text)
            pos_texts.append(pos_text)
        if self.model_backbone_name not in process_vlm_inputs_fns:
            raise ValueError(f"Unsupported model backbone '{self.model_backbone_name}' for VLM processing.")

        process_fn = functools.partial(
            process_vlm_inputs_fns[self.model_backbone_name],
            processor=self.processor,
            max_length=self.data_args['max_len']
        )
        pos_batch_encoding = process_fn({'text': pos_texts, 'images': pos_pil_images})
        qry_batch_encoding = process_fn({'text': qry_texts, 'images': qry_pil_images})
        for batch_encoding in [qry_batch_encoding, pos_batch_encoding]:
            if 'input_ids' in batch_encoding:
                batch_encoding['input_ids'] = self._ensure_tensor_and_type(batch_encoding['input_ids'], torch.long)

            if 'pixel_values' in batch_encoding and isinstance(batch_encoding['pixel_values'], list):
                tensors = [torch.as_tensor(img).float() if isinstance(img, np.ndarray) else img.float()
                           for img in batch_encoding['pixel_values']]
                # batch_encoding['pixel_values'] = torch.stack(tensors, dim=0)

            if 'attention_mask' in batch_encoding:
                batch_encoding['attention_mask'] = self._ensure_tensor_and_type(batch_encoding['attention_mask'],
                                                                                torch.long)
        if ids and ids[0] == '-':
            raise AssertionError("wrong ids, please check your ids")

        return qry_batch_encoding, pos_batch_encoding, ids, sample_ids

@dataclass
class EvaTextImageDataCollator:
    data_args: object
    model_args: object
    processor: object
    model_backbone_name: str

    def _stack_pixel_values(self, pixel_values_list, reference_input_ids_for_batch_size):
        """
        Helper function to stack a list of NumPy arrays/Nones into a PyTorch tensor.
        """
        if not any(pv is not None for pv in pixel_values_list):  # All are None
            return None

        res = torch.stack(pixel_values_list, dim=0)

        return res

    def _ensure_tensor_and_type(self, data: Union[List[List[int]], List[int], torch.Tensor],
                                dtype: torch.dtype) -> torch.Tensor:

        if type(data) is list:
            if all(type(x) is list for x in data):
                tensor_list = [torch.tensor(x, dtype=dtype) for x in data]
                return torch.stack(tensor_list)
            elif all(type(x) is int or type(x) is float for x in data):
                return torch.tensor(data, dtype=dtype)
            else:
                raise TypeError(
                    f"Unexpected list content in _ensure_tensor_and_type: elements are of type {type(data[0])}")

        elif type(data) is torch.Tensor:
            return data.to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)

    def _get_image(self, img_path):
        if not type(img_path) is str or img_path.split('/')[-1] == "-":
            return None
        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args['model_backbone']
        if backbone != PHI3V and self.model_args['image_size']:
            return process_image(image, self.model_args['image_size'])
        else:
            return image

    def __call__(self, examples):
        qry_texts, qry_pil_images, pos_texts, pos_pil_images = [], [], [], []
        building_ids = []
        raw_images =[]
        for ex in examples:
            qry_img_data = ex.get("qry_img_path")
            pos_img_data = ex.get("pos_img_path")
            qry_pil_images.append(qry_img_data)
            pos_pil_images.append(pos_img_data)
            raw_images.append(pos_img_data)
            qry_text= ex.get("qry_text", "")
            pos_text = ex.get("pos_text", "")
            if qry_img_data is not None and isinstance(qry_img_data, Image.Image) and qry_text=="-":
                qry_text = vlm_image_tokens[self.model_args['model_backbone']]
            if pos_img_data is not None and isinstance(pos_img_data, Image.Image) and pos_text=="-":
                pos_text = vlm_image_tokens[self.model_args['model_backbone']]
            qry_text = "<|endoftext|>"+qry_text
            pos_text =  "<|endoftext|>"+ pos_text
            qry_texts.append(qry_text)
            pos_texts.append(pos_text)

            id = ex.get("building_id")
            building_ids.append(id)



        if self.model_backbone_name and self.model_backbone_name not in process_vlm_inputs_fns:
            raise ValueError(f"Unsupported model backbone '{self.model_backbone_name}' for VLM processing.")

        process_fn = functools.partial(process_vlm_inputs_fns[self.model_backbone_name],
                                            processor=self.processor, max_length=self.data_args['max_len'])

        if self.data_args["shuffle"]:
            shuffled_queries = []
            for qry in qry_texts:
                text_list = [s.strip().rstrip('.') for s in re.split(r'\. ', qry) if s.strip()]
                if len(text_list) > 1:
                    random.shuffle(text_list)
                new_qry = ". ".join(text_list)
                if new_qry:
                    new_qry += "."
                shuffled_queries.append(new_qry)
            qry_texts = shuffled_queries


        qry_raw_batch_inputs = {'text': qry_texts, 'images': qry_pil_images}
        pos_raw_batch_inputs = {'text': pos_texts, 'images': pos_pil_images}


        qry_batch_encoding = process_fn(qry_raw_batch_inputs)
        pos_batch_encoding = process_fn(pos_raw_batch_inputs)

        if self.data_args['dataset_name'] == 'GeoText':
            building_ids = np.array(building_ids)

        for batch_encoding in [qry_batch_encoding, pos_batch_encoding]:
            if 'input_ids' in batch_encoding:
                batch_encoding['input_ids'] = self._ensure_tensor_and_type(batch_encoding['input_ids'], torch.long)

            if 'pixel_values' in batch_encoding and isinstance(batch_encoding['pixel_values'], list):
                tensors = [torch.as_tensor(img).float() if isinstance(img, np.ndarray) else img.float()
                           for img in batch_encoding['pixel_values']]
                # batch_encoding['pixel_values'] = torch.stack(tensors, dim=0)

            if 'attention_mask' in batch_encoding:
                batch_encoding['attention_mask'] = self._ensure_tensor_and_type(batch_encoding['attention_mask'],
                                                                                torch.long)


        if "GeoText" in self.data_args['json_path']:
            return qry_batch_encoding, pos_batch_encoding, building_ids, qry_texts, raw_images
        elif "CVGText" in self.data_args['json_path']:
            return qry_batch_encoding, pos_batch_encoding, building_ids,None,None


class GeoTrainTextImageDataset(Dataset):
    def __init__(self, data_args, model_args,\
                 qry_text_field, qry_img_path_field,pos_text_field, pos_img_path_field,\
                 batch_size, cli_args):
        self.data_args = data_args
        self.model_args = model_args
        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field
        self.batch_size = batch_size


        with open(data_args['json_path'], "r", encoding='utf-8') as f:
            print("data_args['json_path']:",data_args['json_path'])
            self.train_data = json.load(f)
        self.samples = copy.deepcopy(self.train_data)
        self.qry_paired_data = self.get_paired_data(qry_text_field, qry_img_path_field)
        self.pos_paired_data = self.get_paired_data(pos_text_field, pos_img_path_field)

    def __len__(self):
        return len(self.samples)

    def _get_image(self, img_path):
        if not type(img_path) is str or img_path.split('/')[-1] == "-":
            return None
        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        image = Image.open(full_img_path).convert("RGB")
        backbone = self.model_args['model_backbone']
        if backbone != PHI3V and self.model_args['image_size']:
            image = process_image(image, self.model_args['image_size'])
            return image
        else:
            return image

    def __getitem__(self, data_idx):
        if type(data_idx) is int:
            cur = self.samples[data_idx].copy()
        else:
            raise TypeError(f"Dataset should only handle integer index. Got: {type(data_idx)}")

        if self.model_args['model_backbone'] != SMOLVLM:

            if "<image>" in cur[self.qry_text_field]:
                cur[self.qry_text_field] = cur[self.qry_text_field].replace("<image>", vlm_image_tokens[self.model_args['model_backbone']])
            elif cur[self.qry_text_field] == '-':
                cur[self.qry_text_field] = vlm_image_tokens[self.model_args['model_backbone']]

            if "<image>" in cur[self.pos_text_field]:
                cur[self.pos_text_field] = cur[self.pos_text_field].replace("<image>", vlm_image_tokens[self.model_args['model_backbone']])
            elif cur[self.pos_text_field] == '-':
                cur[self.pos_text_field] = vlm_image_tokens[self.model_args['model_backbone']]

        # processing pos_text_field
        if (cur[self.pos_text_field] == '-' and cur[self.pos_img_path_field] != '-') or \
                (cur[self.pos_text_field] == vlm_image_tokens[self.model_args['model_backbone']] and cur[self.pos_img_path_field] != '-'):
            cur[self.pos_text_field] = vlm_image_tokens[self.model_args['model_backbone']]
            cur['building_id'] = self.pos_paired_data[data_idx]['building_id']

        elif (cur[self.qry_text_field] == '-' and cur[self.qry_img_path_field] != '-') or \
                (cur[self.qry_text_field] == vlm_image_tokens[self.model_args['model_backbone']] and cur[self.qry_img_path_field] != '-'):
            cur[self.qry_text_field] = vlm_image_tokens[self.model_args['model_backbone']]
            cur['building_id']= self.qry_paired_data[data_idx]['building_id']


        cur[self.qry_img_path_field] = self._get_image(cur[self.qry_img_path_field])
        cur[self.pos_img_path_field] = self._get_image(cur[self.pos_img_path_field])
        return cur, data_idx


    def shuffle(self):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        my_print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.samples)
        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []
        # counter
        break_counter = 0
        pbar = tqdm()

        while True:

            pbar.update()
            if len(pair_pool) > 0:
                # print("enter if")
                pair = pair_pool.pop(0)
                idx = None
                image_path = None
                qry_text, qry_img_path = pair[self.qry_text_field],pair[self.qry_img_path_field]
                pos_text, pos_img_path = pair[self.pos_text_field],pair[self.pos_img_path_field]
                if pos_img_path=='-':
                    idx = qry_img_path.split("/")[-2]
                    image_path = qry_img_path
                elif qry_img_path=='-':
                    idx = pos_img_path.split("/")[-2]
                    image_path = pos_img_path
                else:
                    assert "not match img path"

                if idx not in idx_batch and image_path not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(image_path)
                    break_counter = 0

                else:
                    if image_path not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)

        self.samples = batches
        my_print("batches:",len(batches))
        my_print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_data), len(self.samples)))
        my_print("Break Counter:", break_counter)
        my_print("Pairs left out of last batch to avoid creating noise:", len(self.train_data) - len(self.samples))

    # def get_paired_data(self, text_field, img_path_field):
    #     """
    #     (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
    #     """
    #     my_print("enter pair")
    #     unique_pair = set()
    #     for row in self.samples:
    #         print("row:",row)
    #         text = ''
    #         if type(row[text_field]) is str:
    #             print("is str")
    #             if row["qry_img_path"] !='-':
    #                 building_id = row["qry_img_path"].split("/")[-2]
    #             else:
    #                 building_id = row["pos_img_path"].split("/")[-2]
    #
    #             if row[text_field]:
    #                 unique_pair.add((row[text_field], row[img_path_field],building_id))
    #             else:
    #                 if type(row[img_path_field]) is List:
    #                     for img_path in row[img_path_field]:
    #                         unique_pair.add((row[text_field], img_path,building_id))
    #                 else:
    #                     unique_pair.add((row[text_field], row[img_path_field],building_id))
    #         elif type(row[text_field]) == list and type(row[text_field][0]) is str and type(row[img_path_field]) is str:
    #             if row["qry_img_path"] !='-':
    #                 building_id = row["qry_img_path"].split("/")[-2]
    #             else:
    #                 building_id = row["pos_img_path"].split("/")[-2]
    #             for t in row[text_field]:
    #                 text+=t
    #             unique_pair.add((text, row[img_path_field],building_id))
    #
    #     paired_data = [{"text": text, "img_path": img_path, "building_id": building_id} for text, img_path, building_id in unique_pair]
    #
    #     my_print("done pair")
    #     return paired_data
    def get_paired_data(self, text_field, img_path_field):

        paired_data = []

        for row in self.samples:
            if isinstance(row[text_field], str):
                if row["qry_img_path"] != '-':
                    building_id = row["qry_img_path"].split("/")[-2]
                else:
                    building_id = row["pos_img_path"].split("/")[-2]

                if row[text_field]:
                    paired_data.append(
                        {"text": row[text_field], "img_path": row[img_path_field], "building_id": building_id})
                else:
                    if isinstance(row[img_path_field], list):
                        for img_path in row[img_path_field]:
                            paired_data.append(
                                {"text": row[text_field], "img_path": img_path, "building_id": building_id})
                    else:
                        paired_data.append(
                            {"text": row[text_field], "img_path": row[img_path_field], "building_id": building_id})

            elif isinstance(row[text_field], list) and isinstance(row[text_field][0], str) and isinstance(
                    row[img_path_field], str):
                if row["qry_img_path"] != '-':
                    building_id = row["qry_img_path"].split("/")[-2]
                else:
                    building_id = row["pos_img_path"].split("/")[-2]

                for t in row[text_field]:
                    paired_data.append({"text": t, "img_path": row[img_path_field], "building_id": building_id})

        return paired_data


class GeoEvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, qry_text_field, qry_img_path_field, pos_text_field, pos_img_path_field,backbone):
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = backbone
        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field

        # # Load local dataset
        # if 'train' in data_args['json_path']:
        #     if 'geo_text2drone' not in data_args['json_path']:
        #         data_args['json_path'] =data_args['json_path'].replace(data_args['json_path'].split('/')[-1], 'geo_text2drone_eva.json')
        json_path = os.path.join(data_args['json_path'])
        my_print(f"Loading evaluation data from: {json_path}")
        try:
            with open(json_path, "r") as f:
                self.eval_data = json.load(f)
        except FileNotFoundError:
            my_print(f"Error: JSON file not found at {json_path}")
            self.eval_data = []

        # Process and store data in a list of dictionaries
        self.samples = self._get_samples()
        my_print(f"Loaded {len(self.samples)} samples for subset: {subset}")

    def _get_samples(self):
        unique_pairs = set()
        samples_list = []

        for row in self.eval_data:
            qry_text_data = row.get(self.qry_text_field, "")
            qry_img_path_data = row.get(self.qry_img_path_field, "-")
            pos_text_data = row.get(self.pos_text_field, "")
            pos_img_path_data = row.get(self.pos_img_path_field, "-")

            path_to_check = None
            if qry_img_path_data and qry_img_path_data != "-":
                if qry_text_data == "-":
                    qry_text_data = vlm_image_tokens[self.model_args['model_backbone']]
                path_to_check = qry_img_path_data
            elif pos_img_path_data and pos_img_path_data != "-":
                if pos_text_data == "-":
                    pos_text_data = vlm_image_tokens[self.model_args['model_backbone']]
                path_to_check = pos_img_path_data
            else:
                assert "something went wrong"

            # path_to_check = qry_img_path_data if qry_img_path_data != '-' else (
            #     pos_img_path_data if pos_img_path_data != '-' else None)

            building_id = path_to_check.split('/')[-2]

            pair = (qry_text_data,pos_text_data, qry_img_path_data, pos_img_path_data,building_id)

            if pair not in unique_pairs:
                samples_list.append({
                    "qry_text": qry_text_data,
                    "qry_img_path": qry_img_path_data,
                    "pos_text": pos_text_data,
                    "pos_img_path": pos_img_path_data,
                    "building_id": building_id
                })
                unique_pairs.add(pair)


        return samples_list

    def __len__(self):
        return len(self.samples)

    def _get_image(self, img_path):
        if not isinstance(img_path, str) or img_path == "" or img_path == '-':
            return None

        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        try:
            image = Image.open(full_img_path).convert("RGB")  # Ensure RGB
        except FileNotFoundError:
            my_print(f"Warning: Image file not found at {full_img_path}. Returning None.")
            return None
        except Exception as e:
            my_print(f"Warning: Could not load image {full_img_path}. Error: {e}. Returning None.")
            return None

        # Apply resolution processing if needed (excluding PHI3V or if no resolution is set)
        if self.backbone and self.backbone != PHI3V and self.model_args['image_size']:
            return process_image(image, self.model_args['image_size'])
        else:
            return image

    def __getitem__(self, data_idx):
        if type(data_idx) is int:
            cur = self.samples[data_idx].copy()

            cur[self.qry_img_path_field], cur[self.pos_img_path_field] = self._get_image(cur[self.qry_img_path_field]), \
                self._get_image(cur[self.pos_img_path_field])



        elif type(data_idx) is List[int] or type(data_idx) is np.ndarray:
            cur = []
            for idx in data_idx:
                cur_ = self.samples[idx].copy()
                cur_[self.qry_img_path_field], cur_[self.pos_img_path_field] = self._get_image(
                    cur_[self.qry_img_path_field]), \
                    self._get_image(cur_[self.pos_img_path_field])
                cur.append(cur_)
        else:
            raise TypeError(f"Unsupported index type: {type(data_idx)}")

        if 'SmolVLM' in self.model_args['model_name_or_path']:
            if "<|image_1|>" in cur[self.qry_text_field]:
                cur[self.qry_text_field] = cur[self.qry_text_field].replace("<|image_1|>", "<image>")
            elif "<|image_1|>" in cur[self.pos_text_field]:
                cur[self.pos_text_field] = cur[self.pos_text_field].replace("<|image_1|>", "<image>")

        return cur

