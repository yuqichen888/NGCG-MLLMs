import torch
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from src.vlm_backbone.smolvlm import SmolVLMForConditionalGeneration
from src.vlm_backbone.internvl3_5 import InternVLForConditionalGeneration
import math
import numpy as np
import pytorch_lightning as pl
from transformers.image_utils import ChannelDimension
from transformers import AutoProcessor, AutoConfig, AutoModelForVision2Seq, Idefics3ForConditionalGeneration, \
    AutoVideoProcessor
from pytorch_lightning.utilities import rank_zero_only

from src.vlm_backbone.internvl3_5.processing_internvl import InternVLProcessor
from transformers import AutoImageProcessor, AutoVideoProcessor, AutoTokenizer
from transformers import AutoTokenizer


INTERNVL = "internvl"
SMOLVLM = 'smolvlm' # Matches model_args['model_backbone']
IDEFICS3 = 'idefics3'


MODEL2BACKBONE = {
    'internvl': INTERNVL,
    'smolvlm':SMOLVLM,
    'idefics3': IDEFICS3,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

vlm_image_tokens = {
    INTERNVL: "<IMG_CONTEXT>",
    SMOLVLM: "<image>",
    IDEFICS3: "<image>",
}

backbone2model = {
    INTERNVL: InternVLForConditionalGeneration,
    SMOLVLM: SmolVLMForConditionalGeneration,
    IDEFICS3: SmolVLMForConditionalGeneration,
}



@rank_zero_only
def my_print(*args, **kwargs):
    print(*args, **kwargs)

IMAGE_FACTOR = 28
MIN_PIXELS = 128*128
MAX_PIXELS = 336*336
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        my_print(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def load_processor(model_args,args_cli):
    """
    Load processor based on VLM backbone.
    """
    # print("model_args['processor_name']:", model_args['processor_name'])
    if model_args['model_name_or_path']==None:
        my_print("model is none")

    model_name = model_args['model_name_or_path']
    if model_args['model_backbone'] == INTERNVL:
        # "OpenGVLab/InternVL3_5-1B-HF"
        local_path = '/gpfs2/scratch/ychen57/code/LightVec/src/vlm_backbone/internvl3_5/'
        model_name = model_args['model_name_or_path']
        print("model_args['model_name_or_path']:",model_args['model_name_or_path'])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = InternVLProcessor.from_pretrained(pretrained_model_name_or_path=local_path, \
                                                      tokenizer=tokenizer,\
                                                    local_files_only=True)


    elif model_args['model_backbone'] == SMOLVLM or model_args['model_backbone'] == IDEFICS3:
        my_print("Manually loading SmolVLM processor components...")
        if '256' in model_args['model_name_or_path']:
            from src.vlm_backbone.smolvlm.processing_smolvlm import SmolVLMProcessor
            from src.vlm_backbone.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor
            from src.vlm_backbone.smolvlm.video_processing_smolvlm import SmolVLMVideoProcessor

            model_path = '/gpfs2/scratch/ychen57/code/LightVec/src/vlm_backbone/smolvlm/'
        elif '500' in model_args['model_name_or_path']:
            from src.vlm_backbone.smolvlm500.processing_smolvlm import SmolVLMProcessor
            from src.vlm_backbone.smolvlm500.image_processing_smolvlm import SmolVLMImageProcessor
            from src.vlm_backbone.smolvlm500.video_processing_smolvlm import SmolVLMVideoProcessor

            model_path = '/gpfs2/scratch/ychen57/code/LightVec/src/vlm_backbone/smolvlm500/'
        try:


            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            image_processor = SmolVLMImageProcessor.from_pretrained(model_path, local_files_only=True)
            image_processor.do_image_splitting = False

            video_processor = SmolVLMVideoProcessor.from_pretrained(model_path, local_files_only=True)

            processor = SmolVLMProcessor.from_pretrained(
                pretrained_model_name_or_path = model_path,
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor
            )

        except Exception as e:

            my_print(f"Error during manual loading of SmolVLM components: {e}")
            raise

        except Exception as e:
            my_print(f"Error loading AutoProcessor for SmolVLM: {e}")
            my_print("Please ensure the model identifier is correct and the model has a processor resolvable by AutoProcessor.")
            raise

    else:
        my_print("enter else")

        processor = AutoProcessor.from_pretrained(
            model_args['processor_name'] if model_args['processor_name'] else model_args['model_name_or_path'],
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config):
    print("hf_config.model_type:",hf_config.model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):

    input_ids, pixel_values, image_grid_thw = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False

    # -------- 1. text-only --------
    if not visual_inputs or visual_inputs is None or (
        isinstance(visual_inputs, list) and any(i is None for i in visual_inputs)
    ):
        for text in texts:
            if "<image>" in text:
                text = text.replace("<image>", "")
            inputs = processor(
                text=[text], images=None,
                max_length=max_length,
                truncation=True
            )

            # flatten input_ids
            input_id = inputs["input_ids"]
            if isinstance(input_id, list) and isinstance(input_id[0], list):
                input_id = input_id[0]
            elif torch.is_tensor(input_id):
                input_id = input_id.squeeze().tolist()
            input_ids.append(input_id)

            pixel_values.append(None)
            image_grid_thw.append(None)

    # -------- 2. has image --------
    else:
        for text, images in zip(texts, visual_inputs):
            if text is None or text == '-' and images !="-":
                text = f"{vlm_image_tokens[INTERNVL]}"
            if "<image>" in text:
                text = text.replace("<image>", "")
            image_exists = True
            if vlm_image_tokens[INTERNVL] in text:
                inputs = processor(
                    text=[text], images=[images],
                    max_length=max_length,
                    truncation=True
                )
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")

            # flatten input_ids
            input_id = inputs["input_ids"]
            if isinstance(input_id, list) and isinstance(input_id[0], list):
                input_id = input_id[0]
            elif torch.is_tensor(input_id):
                input_id = input_id.squeeze().tolist()
            input_ids.append(input_id)

            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
            else:
                pixel_values.append(None)

            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])
            else:
                image_grid_thw.append(None)

    # -------- 3. padding input_ids --------
    batch_encoding = processor.tokenizer.pad(
        {'input_ids': input_ids},
        return_tensors="pt"
    )
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
    }
    # -------- 4. padding pixel_values --------
    if image_exists:
        pixel_shape = None
        for v in pixel_values:
            if v is not None:
                pixel_shape = np.array(v).shape
                break
        if pixel_shape is None:
            raise ValueError("No valid pixel values found!")

        # converted into tensor.
        # If None then filled with zeros
        pixel_values_tensor = [
            torch.from_numpy(np.array(v)).float() if v is not None else torch.zeros(pixel_shape, dtype=torch.float32)
            for v in pixel_values
        ]
        pixel_values = torch.stack(pixel_values_tensor, dim=0)
        if pixel_values.shape[1] ==1:
            pixel_values = pixel_values.squeeze(1)

        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw
    else:
        inputs['pixel_values'] = None
        inputs['image_grid_thw'] = [None] * input_ids.shape[0]

    return inputs

def SmolVLM_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    # my_print("enter qwen2_VL_process_fn")
    if not visual_inputs or visual_inputs is None or (type(visual_inputs) == list and any(i is None for i in visual_inputs)):
        # all images must be valid
        for text in texts:
            if "<image>" in text:
                text.replace("<image>", "")
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)

            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)

            pixel_values.append(None)

    else:
        for text, images in zip(texts, visual_inputs):
            if text is None or text == '-':
                text = f"{vlm_image_tokens[SMOLVLM]}"

            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if vlm_image_tokens[SMOLVLM] in text:
                ###only feed image token in text -> why padding is False
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=max_length, truncation=True)

            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")

            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
            else:
                pixel_values.append(None)

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")

    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
    }
    if image_exists:
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        # make the batch full tensors
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in
                        pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        # image_grid_thw = np.concatenate(image_grid_thw, axis=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
    else:
        # bs, num_images, num_channels, height, width
        # inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['pixel_values'] = None

    if isinstance(inputs['input_ids'], np.ndarray):
        inputs['input_ids'] = torch.from_numpy(inputs['input_ids']).long()
    if isinstance(inputs['attention_mask'], np.ndarray):
        inputs['attention_mask'] = torch.from_numpy(inputs['attention_mask']).long()

    return inputs


# --- 8. Update process_vlm_inputs_fns dictionary ---
process_vlm_inputs_fns = {
    SMOLVLM: SmolVLM_process_fn,
    INTERNVL:InternVL_process_fn,
    IDEFICS3: SmolVLM_process_fn
}