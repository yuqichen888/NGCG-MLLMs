# evaluate.py
import argparse
import os
import yaml

import numpy as np
import gc
from src.model import MMEBModel, apply_attention_redistribution_for_language
from src.data.geotext import GeoEvalDataset, EvaTextImageDataCollator
from src.data.cvgtext import CVGEvalDataset
from src.data.university import University1652
from src.model_utils import get_backbone_name, load_processor
from distutils.util import strtobool
import torch
from torch.utils.data import DataLoader,random_split
import pytorch_lightning as pl
import torch.serialization
import re

num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

def check_matrix(mat, name):
    if np.any(np.isinf(mat)):
        print(f"WARNING: {name} contains inf!")
    if np.any(np.isnan(mat)):
        print(f"WARNING: {name} contains NaN!")
    if np.max(np.abs(mat)) > 1e10:
        print(f"WARNING: {name} has extreme values (max={np.max(mat)})")


def get_pred_batched(qry_emb, pos_emb, all_gallery_ids, qry_batch_size=1000):
    check_matrix(qry_emb, "qry_emb")
    check_matrix(pos_emb, "pos_emb")
    print("Matrix check done.")
    num_qry = qry_emb.shape[0]
    num_pos = pos_emb.shape[0]
    qry_batch_size = min(qry_batch_size, num_qry)
    all_preds_indices = []
    pos_emb_T = pos_emb.T

    for i in range(0, num_qry, qry_batch_size):
        qry_batch = qry_emb[i:i + qry_batch_size]
        scores_batch = np.dot(qry_batch, pos_emb_T)
        preds_batch_indices = np.argsort(scores_batch, axis=1)[:, ::-1]
        all_preds_indices.append(preds_batch_indices)

        del qry_batch
        del scores_batch
        del preds_batch_indices
        gc.collect()

    preds_final_indices = np.vstack(all_preds_indices)

    del all_preds_indices
    gc.collect()

    final_preds_ids = all_gallery_ids[preds_final_indices]

    del preds_final_indices
    gc.collect()

    return None, None, final_preds_ids

def compute_recall_at_k(cur_query_ids: np.ndarray, preds_id: np.ndarray, ks: tuple = (1, 5, 10)) -> dict:
    recalls = {f'R@{k}': 0.0 for k in ks}
    num_samples = cur_query_ids.shape[0]

    for i in range(num_samples):

        current_query_id = cur_query_ids[i]
        current_gallery_id = preds_id[i]
        # print("current_query_id:", current_query_id)
        # print("current_gallery_id:", current_gallery_id)
        for k_val in ks:
            if current_query_id in current_gallery_id[:k_val]:
                recalls[f'R@{k_val}'] += 1

    for k_val in ks:
        recalls[f'R@{k_val}'] = (recalls[f'R@{k_val}'] / num_samples) * 100.0
    return recalls



class EvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_args: dict, model_args: dict, args_cli: dict, processor, model_backbone: str,
                 eval_batch_size: int, num_workers: int):
        super().__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.args_cli = args_cli
        self.processor = processor
        self.model_backbone = model_backbone
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        if  "GeoText" in self.args_cli.dataset_name:
            self.eval_dataset = GeoEvalDataset(
                self.data_args, self.model_args,
                subset=self.data_args['subset_name'],
                backbone=self.model_backbone,
                qry_text_field="qry_text", qry_img_path_field="qry_img_path",
                pos_text_field="pos_text", pos_img_path_field="pos_img_path"
            )
        elif "CVGText" in self.args_cli.dataset_name:

            print("cvgtext,start CVGEvalDataset.")
            self.eval_dataset = CVGEvalDataset(
                self.data_args, self.model_args,
                subset=self.data_args['subset_name'],
                backbone=self.model_backbone,
                qry_text_field="qry_text", qry_img_path_field="qry_img_path",
                pos_text_field="pos_text", pos_img_path_field="pos_img_path"
            )

        elif "University" in self.args_cli.dataset_name:
            self.eval_dataset = University1652("/gpfs2/scratch/ychen57/Datasets/University-Release/", qry_view='drone', pos_view='satellite', mode='test')
        else:
            raise NotImplementedError

        if self.model_backbone:  # zero-shot clip without model_backbone and it would be None
            self.eval_collator = EvaTextImageDataCollator(
                self.data_args, self.model_args,
                processor=self.processor,
                model_backbone_name=self.model_backbone
            )
    def val_dataloader(self) -> DataLoader:
        if self.args_cli.dataset_name == "GeoText":
            val_size = int(len(self.eval_dataset) * 0.1)
        elif self.args_cli.dataset_name == "CVGText":
            val_size = int(len(self.eval_dataset) * 1.0)
        test_size = len(self.eval_dataset)-val_size

        g = torch.Generator().manual_seed(42)
        val_subset, _ = random_split(self.eval_dataset, [val_size, test_size], generator=g)


        return DataLoader(
            val_subset,
            batch_size=self.eval_batch_size,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            shuffle=False
        )



def main(args_cli):

    pl.seed_everything(1234)

    with open(args_cli.config_path, 'r') as f:
        config = yaml.safe_load(f)
    #
    model_args = config.get('model_args', {})
    data_args = config.get('data_args', {})
    training_args = config.get('training_args', {})

    print("model_args:", model_args)
    data_args["shuffle"] = args_cli.shuffle
    if args_cli.dataset_name is not None:
        data_args["dataset_name"] = args_cli.dataset_name
        if "GeoText" in args_cli.dataset_name:
            data_args["image_dir"] = "/gpfs2/scratch/ychen57/Datasets/" + data_args["dataset_name"] + "/"
            data_args['json_path'] = '/gpfs2/scratch/ychen57/Datasets/GeoText/'

        elif "CVGText" in args_cli.dataset_name:
            data_args["image_dir"] = "/gpfs2/scratch/ychen57/Datasets/" + data_args["dataset_name"]+"/"
            data_args['json_path'] = '/gpfs2/scratch/ychen57/Datasets/CVGText/'

    if args_cli.subset_name is not None:
        data_args["subset_name"] = str(args_cli.subset_name)
        data_args['json_path'] = data_args['json_path'] + data_args['subset_name'] + '_eva.json'
    if args_cli.lora is not None:
        model_args["lora"] = args_cli.lora



    if args_cli.lora_exclude_modules is not None:
        model_args["lora_exclude_modules"] = args_cli.lora_exclude_modules
    slurm_job_name = args_cli.job_name_from_slurm
    if args_cli.lora_alpha:
        model_args['lora_alpha'] = int(args_cli.lora_alpha)

    print("Loading processor...")
    processor = load_processor(model_args,args_cli)
    print("Processor loaded.")

    model_backbone = model_args['model_backbone']
    print(f"Determined model backbone: {model_backbone}")

    eval_batch_size = args_cli.eval_batch_size or data_args.get('batch_size', 10)


    data_module = EvaluationDataModule(
        data_args=data_args,
        model_args=model_args,
        args_cli=args_cli,
        processor=processor,
        model_backbone=model_backbone,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )


    print("model_args:", model_args)

    #
    # model_args['model_name_or_path'] = "/gpfs2/scratch/ychen57/code/LightVec/vlm2vec/lora/"
    #
    # model = MMEBModel.load(model_args, data_args,training_args,args_cli)
    # print("done loaded")
    # model = model.to('cuda', dtype=torch.bfloat16)
    # model.eval()


    model = MMEBModel.build(
        model_args=model_args,
        training_args=training_args,
        data_args=data_args,
        args_cli=args_cli,
        save=args_cli.save,
        is_train = False,
    )

    print(f"Model structure built. Type: {type(model.encoder)}")
    print("model:", model)
    checkpoint = torch.load(args_cli.checkpoint_to_eval, map_location=args_cli.device, weights_only=False)
    state_dict_key = 'state_dict' if 'state_dict' in checkpoint else None

    if args_cli.fea_token == 'query':
        MODULES_TO_IGNORE = []
    else:
        MODULES_TO_IGNORE = [
            r'^qry\_pooling\_module\.',
            r'^pos\_pooling\_module\.',
        ]


    original_state_dict = checkpoint[state_dict_key]
    #
    filtered_state_dict = {}
    keys_removed_count = 0
    keys_kept_count = 0

    for key, value in original_state_dict.items():
        should_keep = True
        for pattern in MODULES_TO_IGNORE:
            if re.match(pattern, key):
                should_keep = False
                keys_removed_count += 1
                break

        if should_keep:
            filtered_state_dict[key] = value
            keys_kept_count += 1

    print("-" * 50)
    print(f"Original keys in Checkpoint: {len(original_state_dict)}")
    print(f"Keys REMOVED (Pooling Modules): {keys_removed_count}")
    print(f"Keys KEPT (Backbone & other): {keys_kept_count}")
    print(f"Successfully filtered state dictionary.")
    print("-" * 50)

    model.load_state_dict(filtered_state_dict, strict=args_cli.strict_load)

    # model.encoder.base_model.vision_tower = act_redistribution(model.encoder.base_model.vision_tower, incre_ratio=args_cli.incre_ratio, num_registers=args_cli.num_registers,is_vision=True, is_train=False)

    if hasattr(model.encoder.base_model, 'language_model'):
        if args_cli.num_registers > 0 and (args_cli.fea_token == 'redis_eos' or args_cli.fea_token == 'redis_two'):
            # base_model.vision_tower = act_redistribution(base_model.vision_tower, incre_ratio=args_cli.incre_ratio, num_registers=args_cli.num_registers,is_vision=True, is_train=is_train)
            apply_attention_redistribution_for_language(model.encoder.language_model, signal_index=[-1])
        elif args_cli.num_registers > 0 and args_cli.fea_token == 'redis_all':
            apply_attention_redistribution_for_language(model.encoder_language_model, signal_index=['all'])

    model.eval()

    if args_cli.device == 'cuda':
        model.cuda()


    print("Model weights loaded from checkpoint.")

    # # print("new load way!!!!!!!!!!")
    # model = MMEBModel.load_from_checkpoint(
    #     checkpoint_path=args_cli.checkpoint_to_eval,
    #     map_location=args_cli.device,
    #     strict=args_cli.strict_load
    # )

    precision_arg = 'bf16-mixed'


    trainer_args = {
        'accelerator': 'gpu' if torch.cuda.is_available() and args_cli.device != 'cpu' else 'cpu',
        'devices': 'auto' if torch.cuda.is_available() and args_cli.device != 'cpu' else 1,
        'precision': precision_arg,
        'default_root_dir': os.path.dirname(args_cli.checkpoint_to_eval),
        'logger': False,
        'enable_progress_bar':True,
        'num_nodes': 1,
    }

    if args_cli.device and args_cli.device.startswith('cuda'):
        trainer_args['accelerator'] = args_cli.device
        trainer_args['devices'] = 1
    elif args_cli.device == 'cpu':
        trainer_args['accelerator'] = 'cpu'
        trainer_args['devices'] = 1

    trainer = pl.Trainer(
        **trainer_args,
    )

    trainer.test(model, datamodule=data_module)

    print("Evaluation complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMEBModel.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to config YAML.')
    parser.add_argument('--checkpoint_to_eval', type=str, required=True, help='Path to .ckpt file.')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Eval batch size (overrides config).')
    parser.add_argument('--eval_output_dir', type=str, default=None, help='Dir for eval results.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device (e.g., "cuda:0", "cpu").')
    parser.add_argument('--strict_load', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Strict state_dict loading.')
    parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Shuffle the text or not.')
    parser.add_argument('--num_registers', type=int, default=0)
    parser.add_argument('--save', type=lambda x: bool(strtobool(x)), default=False, help='save emb or not during testing')
    parser.add_argument('--head_num', type=int, default=1, help='How many head were used in aggregated module for cross attention(1,2,3,4,8)')
    parser.add_argument('--query', type=int, default=1, help='the number of query in cross-attention')
    parser.add_argument('--layer', type=int, default=1, help='the number of layers that will used for cross-attention')

    parser.add_argument('--fea_token', type=str, default='eos', help='using whick token feature for retrieval. choice:[eos, whole, query]')
    parser.add_argument('--fea_dim', type=int, default=0,help='final feature dimension for contrastive loss')

    parser.add_argument('--job_name_from_slurm', type=str, help='get job number from slurm job name')
    parser.add_argument('--subset_name', type=str, default=None, help='subset name of the dataset')
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset name, CVGText or GeoText')
    parser.add_argument('--lora_exclude_modules', type=str, default=None,
                        help='setting which module wants to be full trained')
    parser.add_argument('--lora_alpha', type=int, default=64,help='lora_alpha')

    parser.add_argument('--lora', type=lambda x: bool(strtobool(x)), default=False, help='Use LoRA or not')

    parser.add_argument('--augmentation', type=str, default=None, help='augmentation strategy for image')

    parser.add_argument('--task', type=str, default="none",
                        help='task that want to evaluate.')
    parsed_args_cli = parser.parse_args()
    main(parsed_args_cli)