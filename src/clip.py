import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import CLIPModel
import gc
from peft import LoraConfig, get_peft_model

def check_matrix(mat, name):
    if np.any(np.isinf(mat)):
        print(f"WARNING: {name} contains inf!")
    if np.any(np.isnan(mat)):
        print(f"WARNING: {name} contains NaN!")
    if np.max(np.abs(mat)) > 1e10:
        print(f"WARNING: {name} has extreme values (max={np.max(mat)})")


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
    preds_final_indices = preds_final_indices.astype(int)
    final_preds_ids = all_gallery_ids[preds_final_indices]

    del preds_final_indices
    gc.collect()

    return scores, None, final_preds_ids  # not return scores and preds matrixs，only return the final ID ranking



class CLIP(pl.LightningModule):
    def __init__(self, model_name, lr, temperature=0.07):
        super().__init__()
        self.save_hyperparameters()
        model = CLIPModel.from_pretrained(model_name)
        self.lr = lr
        self.temperature = temperature
        config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none"
        )

        # 3. 将 LoRA 应用于模型
        # 注意：你可以选择只给视觉编码器加，或者全部加
        self.model = get_peft_model(model, config)
    # def forward(self, inputs):
    #     # Extract features for both sides
    #
    #     qry_outputs = self.model.get_text_features(inputs['input_ids'])
    #     pos_outputs = self.model.get_image_features(inputs['pixel_values'])
    #
    #     # Normalize for cosine similarity
    #     qry_reps = F.normalize(qry_outputs, p=2, dim=-1)
    #     pos_reps = F.normalize(pos_outputs, p=2, dim=-1)
    #     return qry_reps, pos_reps
    def forward(self, qry_batch,pos_batch):
        # Extract features for both sides

        if 'input_ids' in qry_batch:
            qry_outputs = self.model.get_text_features(
                input_ids=qry_batch['input_ids'],
                attention_mask=qry_batch.get('attention_mask')
            )
        else:
            qry_outputs = self.model.get_image_features(
                pixel_values=qry_batch['pixel_values']
            )

        if 'pixel_values' in pos_batch:
            pos_outputs = self.model.get_image_features(
                pixel_values=pos_batch['pixel_values']
            )
        else:
            pos_outputs = self.model.get_text_features(
                input_ids=pos_batch['input_ids'],
                attention_mask=pos_batch.get('attention_mask')
            )

        # Normalize for cosine similarity
        qry_reps = F.normalize(qry_outputs, p=2, dim=-1)
        pos_reps = F.normalize(pos_outputs, p=2, dim=-1)
        return qry_reps, pos_reps

    def training_step(self, batch, batch_idx):
        qry,pos, ids, sample_ids = batch

        # Get normalized embeddings
        qry_reps, pos_reps = self(qry, pos)

        # Calculate similarity matrix [Batch, Batch]
        logits = (qry_reps @ pos_reps.t()) / self.temperature

        # Ground truth is the diagonal (batch_idx matches batch_idx)
        labels = torch.arange(len(qry_reps), device=self.device)

        # Symmetric Loss (Text-to-Image and Image-to-Text)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ================= (Validation) =================
    def validation_step(self, batch, batch_idx):
        qry,pos, building_ids, sample_ids = batch

        qry_reps, pos_reps = self(qry,pos)

        # computing Loss
        logits = (qry_reps @ pos_reps.t()) / self.temperature
        labels = torch.arange(len(qry_reps), device=self.device)
        val_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)

        # computing Recall@K
        output = {
            "qry_reps": qry_reps.detach(),
            "pos_reps": pos_reps.detach(),
            "ids": building_ids
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        qry_reps, pos_reps,ids,_= self.validation_step_outputs
        self._evaluate_recall(qry_embeddings=qry_reps,pos_embeddings=pos_reps,ids=ids)
        self.validation_step_outputs.clear()

    # ================= 测试步 (Test) =================
    def test_step(self, batch, batch_idx):
        qry, pos, ids, sample_ids = batch
        qry_reps, pos_reps = self(qry, pos)
        output = {
            "qry_reps": qry_reps.detach(),
            "pos_reps": pos_reps.detach(),
            "ids": ids
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        qry_reps = torch.cat([x["qry_reps"] for x in self.test_step_outputs], dim=0)
        pos_reps = torch.cat([x["pos_reps"] for x in self.test_step_outputs], dim=0)
        ids = []
        for x in self.test_step_outputs:
            ids.extend(x["ids"])
        ids = np.array(ids).flatten()
        self._evaluate_recall(qry_embeddings=qry_reps,pos_embeddings=pos_reps,ids=ids)
        self.test_step_outputs.clear()


    def _evaluate_recall(self, qry_embeddings, pos_embeddings, ids):
        local_qry_emb =qry_embeddings.to(self.device)
        local_pos_emb = pos_embeddings.to(self.device)

        all_qry_tensor_np = local_qry_emb.cpu().numpy()
        all_pos_tensor_np = local_pos_emb.cpu().numpy()

        scores_qry2pos, _, preds_id_qry2pos = get_pred_batched(
            all_qry_tensor_np, all_pos_tensor_np, ids
        )
        res_qry2pos = compute_recall_at_k(ids, preds_id_qry2pos)
        print("result for qry2pos:", res_qry2pos)
        print("==================================")

        scores_pos2qry, _, preds_id_pos2qry = get_pred_batched(
             all_pos_tensor_np, all_qry_tensor_np,ids
        )
        res_pos2qry = compute_recall_at_k(ids, preds_id_pos2qry)
        print("result for pos2qry:", res_pos2qry)
        return res_qry2pos

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)