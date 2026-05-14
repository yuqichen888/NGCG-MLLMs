import torch
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.geotext import GeoTrainTextImageDataset, GeoEvalDataset, TrainTextImageDataCollator,\
    EvaTextImageDataCollator
from src.data.cvgtext import OSMTextImageDataset,CVGValDataset
from src.data.university import University1652
from src.model import MMEBModel
from src.model_utils import load_processor, get_backbone_name
import yaml
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import datetime
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as plg
import os
from distutils.util import strtobool
from eva import EvaluationDataModule
from torch.utils.data.distributed import DistributedSampler
import re
import tempfile
import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
wandb.login(key="961e91328b60ae852b2c13c222d71175ec1ec591")

def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False
def count_parameters(model):
    total_params = 0
    trainable_params = 0
    lora_params = 0
    lines = []

    lines.append("=== Trainable Parameter Report ===\n")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            is_lora = "lora" in name.lower()
            if is_lora:
                lora_params += num_params
                tag = "[LoRA]"
            else:
                tag = "[Trainable]"
            lines.append(f"{tag:10} {name:60} {num_params:12,}")
        else:
            lines.append(f"{'[Frozen]':10} {name:60} {num_params:12,}")

    lines.append("\n=== Summary ===")
    lines.append(f"Total parameters:       {total_params:,}")
    lines.append(f"Trainable parameters:   {trainable_params:,}")
    lines.append(f" - of which LoRA:       {lora_params:,}")
    lines.append(f" - of which full model: {trainable_params - lora_params:,}")

    return "\n".join(lines)

class GeoEvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: dict, model_cfg: dict, processor, model_backbone: str,
                 eval_batch_size: int, num_workers: int):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.processor = processor
        self.model_backbone = model_backbone
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.eval_dataset = GeoEvalDataset(
            self.data_cfg, self.model_cfg,
            subset=self.data_cfg['subset_name'],
            backbone=self.model_backbone,
            qry_text_field="qry_text", qry_img_path_field="qry_img_path",
            pos_text_field="pos_text", pos_img_path_field="pos_img_path"
        )
        self.eval_collator = EvaTextImageDataCollator(
            self.data_cfg, self.model_cfg,
            processor=self.processor,
            model_backbone_name=self.model_backbone
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.eval_collator,
            num_workers=self.num_workers,
            shuffle=False
        )

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_args, model_args,args_cli,
                 backbone_name,
                 processor,
                 qry_text_field="qry_text", qry_img_path_field="qry_img_path",
                 pos_text_field="pos_text", pos_img_path_field="pos_img_path",
                 batch_size: int = 8,
                 num_workers: int = 4):
        super().__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.args_cli = args_cli
        self.qry_text_field = qry_text_field
        self.qry_img_path_field = qry_img_path_field
        self.pos_text_field = pos_text_field
        self.pos_img_path_field = pos_img_path_field
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.data_args['dataset_name'] == 'GeoText':
            full_dataset = GeoTrainTextImageDataset(
                self.data_args, self.model_args,
                self.qry_text_field, self.qry_img_path_field,
                self.pos_text_field, self.pos_img_path_field,
                batch_size=self.batch_size, cli_args=self.args_cli
            )

        elif self.data_args['dataset_name'] == 'CVGText':
            full_dataset = OSMTextImageDataset(
                self.data_args, self.model_args,self.args_cli,
                self.qry_text_field, self.qry_img_path_field,
                self.pos_text_field, self.pos_img_path_field
            )
        elif self.data_args['dataset_name'] == 'University1652':
            full_dataset = University1652("/gpfs2/scratch/ychen57/Datasets/University-Release/", qry_view='drone', pos_view='satellite', mode='train')

        self.train_dataset = full_dataset
        # collator
        self.train_collator = TrainTextImageDataCollator(
            data_args, model_args,args_cli,
            model_backbone_name=backbone_name,
            processor=processor,
            is_train=True
        )

    def train_dataloader(self):
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=sampler,
            collate_fn=self.train_collator,
            num_workers=self.num_workers
        )


def load_and_filter_checkpoint(checkpoint_path, model_instance, modules_to_ignore):
    """
    Loads a checkpoint, filters out specified module keys, and returns the modified state dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda',weights_only=False)
    state_dict_key = 'state_dict'
    if state_dict_key not in checkpoint:
        raise KeyError(f"State dictionary key '{state_dict_key}' not found in checkpoint.")

    original_state_dict = checkpoint[state_dict_key]
    filtered_state_dict = {}
    keys_removed_count = 0

    for key, value in original_state_dict.items():
        is_ignored = False
        for pattern in modules_to_ignore:
            if re.match(pattern, key):
                is_ignored = True
                keys_removed_count += 1
                break

        if not is_ignored:
            filtered_state_dict[key] = value

    print(f"Total keys removed: {keys_removed_count}. Ready to create temporary checkpoint.")

    checkpoint[state_dict_key] = filtered_state_dict

    return checkpoint

def main(args_cli):
    pl.seed_everything(1234)
    path=args_cli.config_path
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    model_args = config.get('model_args', {})
    training_args = config.get('training_args', {})
    data_args = config.get('data_args', {})
    data_args["shuffle"] = args_cli.shuffle
    if args_cli.frozen is not None:
        print(f"Overriding model_args.frozen from CLI: {args_cli.frozen}")
        model_args['frozen'] = args_cli.frozen
    if isinstance(model_args["learning_rate"], str):
        model_args["learning_rate"] = float(model_args["learning_rate"])

    if args_cli.epochs is not None:
        training_args["epochs"] = args_cli.epochs
    if args_cli.batch_size is not None:
        training_args["batch_size"] = args_cli.batch_size
    if args_cli.lr is not None:
        model_args["learning_rate"] = args_cli.lr
    # if isinstance(training_args["max_steps"], str):
    #     training_args["max_steps"] = int(training_args["max_steps"])

    if args_cli.dataset_name is not None:
        data_args["dataset_name"] = args_cli.dataset_name
        if "GeoText" in args_cli.dataset_name :
            data_args["image_dir"] = "/gpfs2/scratch/ychen57/Datasets/" + data_args["dataset_name"]+"/train/"
            data_args['json_path'] = '/gpfs2/scratch/ychen57/Datasets/GeoText/vec/'

        elif "CVGText" in args_cli.dataset_name:
            data_args["image_dir"] = "/gpfs2/scratch/ychen57/Datasets/" + data_args["dataset_name"]
            data_args['json_path'] = '/gpfs2/scratch/ychen57/Datasets/CVGText/'
    if args_cli.subset_name is not None:
        data_args["subset_name"] = str(args_cli.subset_name)
        data_args['json_path'] = data_args['json_path'] + data_args['subset_name'] + '_train.json'
    if args_cli.lora is not None:
        model_args["lora"] = args_cli.lora
        model_args["lora_alpha"] = args_cli.lora_alpha

    dataset_name_part = data_args.get('subset_name')
    save_log_name = str(args_cli.job_name_from_slurm)+"_BS" + str(training_args['batch_size'])+"_LR" + str(model_args["learning_rate"])+"_D" + str(dataset_name_part)


    if 'geotext' in args_cli.augmentation:
        save_log_name = save_log_name + "_Aug-geotext"
    elif 'geometry' in args_cli.augmentation:
        save_log_name = save_log_name + "_Aug-geometry"
    elif 'none' in args_cli.augmentation or 'None' in args_cli.augmentation:
        save_log_name = save_log_name + "_Aug-none"

    if args_cli.temperature is not None:
        if isinstance(args_cli.temperature, str):
            model_args["temperature"] = float(args_cli.temperature)
        model_args["temperature"] = args_cli.temperature
        save_log_name = save_log_name + "_T" + str(args_cli.temperature)

    save_log_name = save_log_name + "_ImgRes-" + str(model_args['image_size'])
    save_log_name = save_log_name

    try:
        processor = load_processor(model_args,args_cli)
        print("Processor loaded successfully. processor:", processor)
    except Exception as e:
        print(f"Error loading processor: {e}")

        return

    if model_args['model_backbone'] == 'smolvlm':
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    elif model_args['model_backbone'] == 'internvl':
        print("processor.tokenizer:",processor.tokenizer)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    elif model_args['model_backbone'] == 'phi3_v':
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_1|>")

    elif 'qwen' in model_args['model_backbone'] :
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print("*** image_token_id ***:", image_token_id)
    model_args['image_token_id'] = image_token_id


    model_backbone = model_args['model_backbone']

    train_module = DataModule(
        data_args=data_args,
        model_args=model_args,
        args_cli = args_cli,
        backbone_name=model_backbone,
        processor=processor,
        batch_size=training_args['batch_size'],
    )


    num_samples = len(train_module.train_dataset)
    print("******* train_module.train_dataset **********:", num_samples)


    save_log_name = save_log_name +"_lr"+str(args_cli.lr)

    slurm_job_name = save_log_name
    save_path = os.path.join(training_args['output_dir'], 'checkpoints', slurm_job_name)

    wandb_logger = plg.WandbLogger(
        project=model_args['model_backbone'] + "_" + data_args['dataset_name'] + "_" + data_args[
            'subset_name'], save_dir=save_path,
        log_model=False, name=slurm_job_name)

    train_dataloader = train_module.train_dataloader()

    eva_module = EvaluationDataModule(
        data_args=data_args,
        model_args=model_args,
        args_cli=args_cli,
        processor=processor,
        model_backbone=model_backbone,
        eval_batch_size=training_args['batch_size'],
        num_workers=num_workers
    )

    if args_cli.dataset_name=='GeoText':
        val_size = int(len(eva_module.eval_dataset) * .1)
    elif args_cli.dataset_name =='CVGText':
        val_size = int(len(eva_module.eval_dataset) * 1.)
    test_size = len(eva_module.eval_dataset) - val_size

    g = torch.Generator().manual_seed(42)
    val_subset, _ = random_split(eva_module.eval_dataset, [val_size, test_size], generator=g)

    val_dataloader = DataLoader(
        val_subset,
        batch_size=args_cli.batch_size,
        collate_fn=eva_module.eval_collator,
        num_workers=num_workers,
        shuffle=False
    )

    if data_args['dataset_name'] == 'CVGText':

        callback_instance = ModelCheckpoint(
            dirpath=save_path,
            filename=save_log_name + "_Epoch-{epoch:02d}",
            save_top_k=1,
            monitor="val/recall@1",
            mode="max",
            save_last=True
        )

    elif data_args['dataset_name'] == 'GeoText':
        callback_instance = ModelCheckpoint(
            dirpath=save_path,
            filename=save_log_name + save_log_name + "_Epoch-{epoch:02d}",
            save_top_k=1,
            monitor="val/recall@1",
            mode="max",
            save_last=True
        )

    total_steps = (training_args['epochs'] * num_samples) / args_cli.batch_size

    training_args['steps'] = int(total_steps)
    training_args['warmup_steps'] = int(total_steps * args_cli.warmup_ratio)
    training_args['steps_per_epoch'] = int(num_samples // training_args["batch_size"])

    if args_cli.device == 'gpu':
        training_args['accelerator'] = "cuda"
    elif args_cli.device == 'cpu':
        training_args['accelerator'] = 'cpu'

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # parameters = count_parameters(model)
    # print(f"Total Parameters: {parameters}")
    if model_args['lora']:
        all_args = {
            'cli_args': vars(args_cli),
            'model_args': dict(model_args),
            'training_args': dict(training_args),
            'data_args': dict(data_args),
            # 'parameter_status': parameters,
            'num_gpus': num_gpus,
        }
    else:
        all_args = {
            'cli_args': vars(args_cli),
            'model_args': dict(model_args),
            'training_args': dict(training_args),
            'data_args': dict(data_args),
            'num_gpus': num_gpus,
        }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"combined_config_{timestamp}.yaml")

    with open(path, 'w') as f:
        yaml.dump(all_args, f, default_flow_style=False)

    print(f"Full config saved to: {save_path}")



        # trainer = pl.Trainer(max_epochs=training_args['epochs'],
        #                      # num_sanity_val_steps=0,
        #                      devices= 2 if torch.cuda.is_available() and args_cli.device != 'cpu' else 1,
        #                      accelerator=training_args['accelerator'],
        #                      default_root_dir=training_args['output_dir'],
        #                      precision='bf16-mixed',
        #                      strategy = DDPStrategy(find_unused_parameters=True),
        #                      gradient_clip_val=1.0,
        #                      gradient_clip_algorithm='norm',
        #                      callbacks=[callback_instance])
    accumulate_grad_batches = 1
    trainer = pl.Trainer(max_epochs=training_args['epochs'],
                         num_sanity_val_steps=2,
                         accumulate_grad_batches = accumulate_grad_batches,
                         devices=1 if torch.cuda.is_available() and args_cli.device != 'cpu' else 1,
                         accelerator=training_args['accelerator'],
                         default_root_dir=training_args['output_dir'],
                         precision='bf16-mixed',
                         gradient_clip_val=1.0,
                         gradient_clip_algorithm='norm',
                         callbacks=[callback_instance],
                         logger=wandb_logger)


    model = MMEBModel.build(model_args=model_args, training_args=training_args, data_args=data_args, args_cli=args_cli,
                            save=False, is_train = True)
    print("Built model\n")
    if args_cli.resume_path is not None:
        CHECKPOINT_PATH = args_cli.resume_path
        if args_cli.fea_token == 'query':
            MODULES_TO_IGNORE = []
        else:
            MODULES_TO_IGNORE = [
                r'^qry\_pooling\_module\.',
                r'^pos\_pooling\_module\.',
            ]

        try:
            modified_checkpoint_data = load_and_filter_checkpoint(
                CHECKPOINT_PATH, model, MODULES_TO_IGNORE
            )
            with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
                TEMP_CKPT_PATH = tmp_file.name
            torch.save(modified_checkpoint_data, TEMP_CKPT_PATH)
            print(f"Temporary filtered checkpoint saved to: {TEMP_CKPT_PATH}")
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=TEMP_CKPT_PATH  \
            )

        except Exception as e:
            print(f"CRITICAL ERROR during checkpoint restoration: {e}")

        finally:
            if 'TEMP_CKPT_PATH' in locals() and os.path.exists(TEMP_CKPT_PATH):
                os.remove(TEMP_CKPT_PATH)
                print(f"Cleaned up temporary checkpoint file: {TEMP_CKPT_PATH}")
    else:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    if args_cli.dataset_name =='CVGText':
        trainer.test(model, datamodule=eva_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/gpfs2/scratch/ychen57/code/LightVec/config_internvl3_5.yaml', help='Path to the config YAML file')
    parser.add_argument('--frozen', type=lambda x: (str(x).lower() == 'true'),  # Converts "true"/"false" string to bool
                        default=None, help='Set model_args.frozen (True/False)')
    parser.add_argument('--job_name_from_slurm', type=str, help='get job number from slurm job name')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_registers', type=int, default=4)

    parser.add_argument('--head_num', type=int, default=1, help='How many head were used in aggregated module for cross attention(1,2,3,4,8)')
    parser.add_argument('--query', type=int, default=1, help='the number of query in cross-attention')
    parser.add_argument('--layer', type=int, default=1, help='the number of layers that will used for cross-attention')

    parser.add_argument('--fea_token', type=str, default='eos', help='using whick token feature for retrieval. choice:[eos, whole, query]')
    parser.add_argument('--fea_dim', type=int, default=0,help='final feature dimension for contrastive loss')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--init', type=str, help='initialization method for register token: zeros or ones')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--resume_path', type=str, default=None,
                        help='the path that you want to resume chechpoint.')
    parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Shuffle the text or not.')

    parser.add_argument('--subset_name', type=str, default=None, help='subset name of the dataset')
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset name, CVGText or GeoText')
    parser.add_argument('--device', type=str, default="cpu", help='Device (e.g., "gpu", "cpu").')
    parser.add_argument('--lora', type=lambda x: bool(strtobool(x)), default=False, help='Use LoRA or not')
    parser.add_argument('--lora_alpha', type=int, default=128, help='lora alpha setting')
    parser.add_argument('--augmentation', type=str, default="none", help='augmentation strategy for image')
    parser.add_argument('--temperature', type=float, default=0.02, help='epochs to train')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps for learning rate')
    # parser.add_argument('--neg_threshold', type=int, default=5, help='neg threshold')
    args_cli = parser.parse_args()
    main(args_cli)


