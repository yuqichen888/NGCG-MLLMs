import os
import json
from PIL import Image
from torch.utils.data import Dataset
import random
import copy
from typing import List, Union
from src.model_utils import PHI3V
import numpy as np
import re
from src.model_utils import PHI3V, vlm_image_tokens, QWEN2_5_VL,QWEN2_VL,SMOLVLM
from src.data.augmentation import geotext_train_transform
def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    elif type(resolution) == int:
        image = image.resize((resolution, resolution))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

class OSMTextImageDataset(Dataset):
    def __init__(self, data_args, model_args, cli_args, qry_text_field, qry_img_path_field,pos_text_field, pos_img_path_field):
        self.image_root_dirs = data_args['image_dir']
        self.json_file_paths = data_args['json_path']



        self.data_args = data_args
        self.model_args = model_args
        self.cli_args = cli_args

        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field

        self.qry_image = []
        self.qry_text = []
        self.pos_image = []
        self.pos_text = []

        self.img2txt = {}
        self.txt2img = {}
        if "aug" in cli_args.subset_name and 'combine' not in cli_args.subset_name:
            # 1. 加载第一个数据集
            ori_json_path = re.sub(r'_aug\d+', '', data_args['json_path'])

            with open(ori_json_path, "r", encoding='utf-8') as f:
                self.train_data = json.load(f)

            # 2. 加载第二个数据集
            aug_json_path = re.sub(r'vec/', '', data_args['json_path'])

            with open(aug_json_path, "r", encoding='utf-8') as f:
                augmented_data = json.load(f)

            # 3. 将两个列表合并
            # 使用 .extend() 方法将第二个列表的元素添加到第一个列表中
            if isinstance(self.train_data, list) and isinstance(augmented_data, list):
                self.train_data.extend(augmented_data)
                print("total train data is:", len(self.train_data))
            else:
                assert "!!!!augmented data is not list,cannot be extended!!!!!"
        else:
            with open(self.json_file_paths, 'r') as f:
                self.train_data = json.load(f)

        # for json_file_path, image_root_dir in zip(self.json_file_paths, self.image_root_dirs):

        """
        for idx, item in enumerate(json_data):
            image_name = os.path.basename(item['bev_image'])
            caption = item['caption']

            self.image.append(os.path.join(image_root_dir, image_name))
            self.text.append(caption)

            self.img2txt[len(self.image) - 1] = [len(self.text) - 1]
            self.txt2img[len(self.text) - 1] = len(self.image) - 1
        """
        self.samples = copy.deepcopy(self.train_data)
        self.qry_paired_data = self.get_paired_data(qry_text_field, qry_img_path_field)
        self.pos_paired_data = self.get_paired_data(pos_text_field, pos_img_path_field)

    def __len__(self):
        return len(self.samples)



    def get_paired_data(self, text_field, img_path_field):

        paired_data = []

        for row in self.samples:
            if isinstance(row[text_field], str):
                if row["qry_img_path"] != '-':
                    building_id = row["qry_img_path"].split("/")[-1]
                else:
                    building_id = row["pos_img_path"].split("/")[-1]

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
                    building_id = row["qry_img_path"].split("/")[-1]
                else:
                    building_id = row["pos_img_path"].split("/")[-1]

                for t in row[text_field]:
                    paired_data.append({"text": t, "img_path": row[img_path_field], "building_id": building_id})

        return paired_data

    def _get_image(self, img_path):
        if not type(img_path) is str or img_path.split('/')[-1] == "-":
            return None
        img_path = img_path.lstrip('/')
        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        image = Image.open(full_img_path).convert("RGB")
        backbone = self.model_args['model_backbone']
        if self.model_args['image_size']:
            image = process_image(image, self.model_args['image_size'])
            return image
        else:
            return image



    def __getitem__(self, data_idx):
        if type(data_idx) is int:
            cur = self.samples[data_idx].copy()
        else:
            raise TypeError(f"Dataset should only handle integer index. Got: {type(data_idx)}")
        if 'phi3_v' in self.model_args['model_backbone']:
            # print("enter")
            if "<image>" in cur[self.qry_text_field]:
                cur[self.qry_text_field] = cur[self.qry_text_field].replace("<image>", vlm_image_tokens[PHI3V])

            if "<image>" in cur[self.pos_text_field]:
                cur[self.pos_text_field] = cur[self.pos_text_field].replace("<image>", vlm_image_tokens[PHI3V])

        elif self.model_args['model_backbone'] == QWEN2_VL or self.model_args['model_backbone'] == QWEN2_5_VL:
            # 处理 qry_text_field
            if "<image>" in cur[self.qry_text_field]:
                cur[self.qry_text_field] = cur[self.qry_text_field].replace("<image>", vlm_image_tokens[QWEN2_VL])


        # 处理 pos_text_field
        if  cur[self.pos_text_field] == '-' and cur[self.pos_img_path_field] != '-':
            cur[self.pos_text_field] = vlm_image_tokens[self.model_args['model_backbone']]
            cur['building_id'] = self.pos_paired_data[data_idx]['building_id']
        elif cur[self.qry_text_field] == '-' and cur[self.qry_img_path_field] != '-':
            cur[self.qry_text_field] = vlm_image_tokens[self.model_args['model_backbone']]
            cur['building_id'] = self.qry_paired_data[data_idx]['building_id']

        cur[self.qry_img_path_field] = self._get_image(cur[self.qry_img_path_field])
        cur[self.pos_img_path_field] = self._get_image(cur[self.pos_img_path_field])


        return cur,data_idx


class CVGValDataset(Dataset):
    def __init__(self, data_args, model_args, args_cli, qry_text_field, qry_img_path_field, pos_text_field,
                 pos_img_path_field):
        """
        Initializes the evaluation dataset.

        Args:
            data_args: Arguments related to the dataset.
            model_args: Arguments related to the model.
            subset: The name of the subset JSON file (e.g., "test_queries", "test_gallery").
            text_field: The key for the text data in the JSON (e.g., "qry_text").
            img_path_field: The key for the image path data in the JSON (e.g., "qry_img_path").
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = model_args['model_backbone']
        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field
        self.subset = args_cli.subset_name

        # Load local dataset
        json_path = os.path.join(data_args['json_path'])

        try:
            with open(json_path, "r") as f:
                self.eval_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            self.eval_data = []

        # Process and store data in a list of dictionaries
        self.samples = self._get_samples()


    def _get_image(self, img_path):
        """
        Loads and processes an image from its path.
        Consistent with the training dataset's image loading.
        """
        if not isinstance(img_path, str) or img_path == "" or img_path == '-':
            return None

        # Construct full path - adjust if 'dataset_split' is needed for eval
        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        try:
            image = Image.open(full_img_path).convert("RGB")  # Ensure RGB
        except FileNotFoundError:
            print(f"Warning: Image file not found at {full_img_path}. Returning None.")
            return None
        except Exception as e:
            print(f"Warning: Could not load image {full_img_path}. Error: {e}. Returning None.")
            return None

        # Apply resolution processing if needed (excluding PHI3V or if no resolution is set)
        if self.backbone != PHI3V and self.data_args.get('image_resolution'):
            return process_image(image, self.model_args['image_size'])
        else:
            return image


    def _get_samples(self):
        unique_pairs = set()
        samples_list = []

        for row in self.eval_data:
            qry_text_data = row.get(self.qry_text_field, "")
            qry_img_path_data = row.get(self.qry_img_path_field, "-")
            pos_text_data = row.get(self.pos_text_field, "")
            pos_img_path_data = row.get(self.pos_img_path_field, "-")

            if qry_img_path_data != '-':
                building_id = qry_img_path_data
            elif pos_img_path_data != '-':
                building_id = pos_img_path_data
            else:
                assert "not set building_id"
            pair = (qry_text_data, pos_text_data, qry_img_path_data, pos_img_path_data, building_id)
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

        return cur



class CVGEvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, qry_text_field, qry_img_path_field, pos_text_field, pos_img_path_field,backbone):
        """
        Initializes the evaluation dataset.

        Args:
            data_args: Arguments related to the dataset.
            model_args: Arguments related to the model.
            subset: The name of the subset JSON file (e.g., "test_queries", "test_gallery").
            text_field: The key for the text data in the JSON (e.g., "qry_text").
            img_path_field: The key for the image path data in the JSON (e.g., "qry_img_path").
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = backbone
        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field

        # Load local dataset
        json_path = os.path.join(data_args['json_path'])
        if 'train' in json_path:
            json_path = json_path.replace('train', 'eva')
        print(f"Loading evaluation data from: {json_path}")
        try:
            with open(json_path, "r") as f:
                self.eval_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            self.eval_data = []

        # Process and store data in a list of dictionaries
        self.samples = self._get_samples()
        print(f"Loaded {len(self.samples)} samples for subset: {subset}")

    def _get_image(self, img_path):
        """
        Loads and processes an image from its path.
        Consistent with the training dataset's image loading.
        """
        if not isinstance(img_path, str) or img_path == "" or img_path == '-':
            return None

        # Construct full path - adjust if 'dataset_split' is needed for eval
        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        try:
            image = Image.open(full_img_path).convert("RGB")  # Ensure RGB
        except FileNotFoundError:
            print(f"Warning: Image file not found at {full_img_path}. Returning None.")
            return None
        except Exception as e:
            print(f"Warning: Could not load image {full_img_path}. Error: {e}. Returning None.")
            return None

        # Apply resolution processing if needed (excluding PHI3V or if no resolution is set)
        if self.model_args['image_size']:
            return process_image(image, self.model_args['image_size'])
        else:
            return image



    def _get_samples(self):
        unique_pairs = set()
        samples_list = []

        for row in self.eval_data:
            qry_text_data = row.get(self.qry_text_field, "")
            qry_img_path_data = row.get(self.qry_img_path_field, "-")
            pos_text_data = row.get(self.pos_text_field, "")
            pos_img_path_data = row.get(self.pos_img_path_field, "-")

            if qry_img_path_data != '-':
                building_id = qry_img_path_data
            elif pos_img_path_data != '-':
                building_id = pos_img_path_data
            else:
                assert "not set building_id"

            pair = (qry_text_data, pos_text_data, qry_img_path_data, pos_img_path_data, building_id)

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


        return cur

class CVGEvalDataset_cross(Dataset):
    def __init__(self, args_cli,data_args, model_args, subset, qry_text_field, qry_img_path_field, pos_text_field, pos_img_path_field,backbone):
        """
        Initializes the evaluation dataset.

        Args:
            args_cli (argparse.Namespace): Arguments from CLI.
            data_args: Arguments related to the dataset.
            model_args: Arguments related to the model.
            subset: The name of the subset JSON file (e.g., "test_queries", "test_gallery").
            text_field: The key for the text data in the JSON (e.g., "qry_text").
            img_path_field: The key for the image path data in the JSON (e.g., "qry_img_path").
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = backbone
        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field

        # Load local dataset
        json_path = os.path.join(data_args['json_path'])
        json_path = json_path.replace(args_cli.source, args_cli.target)
        if 'train' in json_path:
            json_path = json_path.replace('train', 'eva')
        print(f"Loading evaluation data from: {json_path}")
        try:
            with open(json_path, "r") as f:
                self.eval_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            self.eval_data = []

        # Process and store data in a list of dictionaries
        self.samples = self._get_samples()
        print(f"Loaded {len(self.samples)} samples for subset: {subset}")

    def _get_image(self, img_path):
        """
        Loads and processes an image from its path.
        Consistent with the training dataset's image loading.
        """
        if not isinstance(img_path, str) or img_path == "" or img_path == '-':
            return None

        # Construct full path - adjust if 'dataset_split' is needed for eval
        full_img_path = os.path.join(self.data_args['image_dir'], img_path)
        try:
            image = Image.open(full_img_path).convert("RGB")  # Ensure RGB
        except FileNotFoundError:
            print(f"Warning: Image file not found at {full_img_path}. Returning None.")
            return None
        except Exception as e:
            print(f"Warning: Could not load image {full_img_path}. Error: {e}. Returning None.")
            return None

        # Apply resolution processing if needed (excluding PHI3V or if no resolution is set)
        if self.model_args['image_size']:
            return process_image(image, self.model_args['image_size'])
        else:
            return image



    def _get_samples(self):
        unique_pairs = set()
        samples_list = []

        for row in self.eval_data:
            qry_text_data = row.get(self.qry_text_field, "")
            qry_img_path_data = row.get(self.qry_img_path_field, "-")
            pos_text_data = row.get(self.pos_text_field, "")
            pos_img_path_data = row.get(self.pos_img_path_field, "-")

            if qry_img_path_data != '-':
                building_id = qry_img_path_data
            elif pos_img_path_data != '-':
                building_id = pos_img_path_data
            else:
                assert "not set building_id"

            pair = (qry_text_data, pos_text_data, qry_img_path_data, pos_img_path_data, building_id)

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


        return cur