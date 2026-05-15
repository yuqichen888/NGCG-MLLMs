import numpy as np
import gc
import argparse
import os
import json
import shutil
from sentry_sdk.utils import epoch


def check_matrix(mat, name):
    if np.any(np.isinf(mat)):
        print(f"WARNING: {name} contains inf!")
    if np.any(np.isnan(mat)):
        print(f"WARNING: {name} contains NaN!")
    if np.max(np.abs(mat)) > 1e10:  
        print(f"WARNING: {name} has extreme values (max={np.max(mat)})")


def get_pred_full(qry_emb, pos_emb, all_gallery_ids):
    """
    The similarity and ranking of all queries are computed at once without chunking.
    Suitable for use with sufficient memory.

    Parameters:
        qry_emb (np.ndarray): Query embedding, shape (N_qry, D).
        pos_emb (np.ndarray): gallery embedding, shape (N_pos, D).
        all_gallery_ids (np.ndarray): Complete gallery ID array, shape (N_pos,).
    """
    check_matrix(qry_emb, "qry_emb")
    check_matrix(pos_emb, "pos_emb")
    print("Matrix check done.")
    pos_emb_T = pos_emb.T
    scores_qry2pos = np.dot(qry_emb, pos_emb_T)

    del pos_emb_T
    gc.collect()

    # --- 2. computing sorted rank ---
    # shape: [N_qry, N_pos]
    preds_qry2pos = np.argsort(scores_qry2pos, axis=1)

    # Invert the index using [:, ::-1] to get descending rank (highest similarity first)
    preds_qry2pos = preds_qry2pos[:, ::-1]

    # Use the rank index to get the ids from all_gallery_ids
    # final_preds_ids shape: [N_qry, N_pos]
    final_preds_ids = all_gallery_ids[preds_qry2pos]
    gc.collect()

    # Return similarity matrix, ranking index (internal index of gallery), and final ID ranking
    return scores_qry2pos, preds_qry2pos, final_preds_ids

def get_pred_batched(qry_emb, pos_emb, all_gallery_ids, qry_batch_size=1000):
    """
    The similarity and ranking are computed in blocks to optimize memory usage.
    Parameters:
        qry_emb (np.ndarray): Query embedding, shape (N_qry, D).
        pos_emb (np.ndarray): gallery embedding, shape (N_pos, D).
        all_gallery_ids (np.ndarray): Complete gallery ID array, shape (N_pos,).
        qry_batch_size (int): The chunk size of the query embedding.
    """
    check_matrix(qry_emb, "qry_emb")
    check_matrix(pos_emb, "pos_emb")
    print("Matrix check done.")

    num_qry = qry_emb.shape[0]

    all_preds_indices = []  # Store the ranking index for all queries

    pos_emb_T = pos_emb.T
    for i in range(0, num_qry, qry_batch_size):
        qry_batch = qry_emb[i:i + qry_batch_size]

        # calculate the similarity matrix scores
        # Shape: [qry_batch_size, num_pos]
        scores_batch = np.dot(qry_batch, pos_emb_T)
        preds_batch_indices = np.argsort(scores_batch, axis=1)[:, ::-1]
        all_preds_indices.append(preds_batch_indices)

        del scores_batch
        gc.collect()

    # Stack the rank indices of all batches vertically
    # This operation creates a new large array, but this is the last time,
    # and the data is integer indexed, so the memory footprint is relatively small
    preds_final_indices = np.vstack(all_preds_indices)

    # free the intermediate list
    del all_preds_indices
    gc.collect()

    # Use the rank index to get the ids from all_gallery_ids
    final_preds_ids = all_gallery_ids[preds_final_indices]

    del preds_final_indices
    gc.collect()

    return  qry_batch, None,preds_batch_indices, final_preds_ids


def compute_recall_at_k(cur_query_ids: np.ndarray, preds_id: np.ndarray, ks: tuple = (1, 5, 10)) -> dict:
    """
    Calculate the recall Recall@k.
    Parameters:
        cur_query_ids (np.ndarray): The actual query ids, in NumPy array format
        preds_id (np.ndarray): A list of predicted gallery ids in NumPy array format.
        ks (tuple): This stores a list of k-values in Recall@k.
    Returns:
        dict: A dictionary containing the results of Recall@k
    """

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

def vis(sim_qry2pos, checkpoint_to_eval, image_dir, image_file_name, type):
    with open("/gpfs2/scratch/ychen57/Datasets/GeoText/geo_t2i_eva.json", 'r', encoding='utf-8') as f:
        data = json.load(f)


    result_dict = {item['qry_text']: item['pos_img_path'] for item in data}

    top5_indices = np.argsort(-sim_qry2pos, axis=1)[:, :5]  # descending order
    top1_indices = top5_indices[:, 0]
    res_list = []

    for i in range(sim_qry2pos.shape[0]):
        correct_index = i  # ground truth (query i and the corresponding pos i)
        ## Save R@5 correct but R@1 fail example
        # if correct_index in top5_indices[i] and correct_index != top1_indices[i]:
        ## save R@1 CORRECT example
        if type =="fail":
            condition = (correct_index in top5_indices[i] and correct_index != top1_indices[i])
        elif type == "success":
            condition = (correct_index == top1_indices[i])
        if condition:
            # gathering information
            res_list.append({
                "query_id": image_file_name[i],
                "top1_pred": image_file_name[top1_indices[i]],
                "top5_preds": [image_file_name[j] for j in top5_indices[i].tolist()],
                "similarities_top5": sim_qry2pos[i, top5_indices[i]].tolist()
            })

            sample = res_list[-1]
            query_id = sample["query_id"]
            top1_path = sample["top1_pred"]
            top5_list = sample["top5_preds"]

            save_root = \
                checkpoint_to_eval.split(checkpoint_to_eval.split('/')[-1])[0]
            query_folder = os.path.join(save_root, f"sample_{i:05d}")
            os.makedirs(query_folder, exist_ok=True)

            if os.path.exists(os.path.join(image_dir, query_id)):
                shutil.copy(os.path.join(image_dir, query_id),
                            os.path.join(query_folder, "query" + str(query_id.split('/')[-1])))
            else:
                print(f"Warning: query image not found: {query_id}")

            # copy top1
            if os.path.exists(os.path.join(image_dir, top1_path)):
                shutil.copy(os.path.join(image_dir, top1_path),
                            os.path.join(query_folder, "top1.jpg"))
            else:
                print(f"Warning: top1 image not found: {top1_path}")

            # copy top5
            for rank, path in enumerate(top5_list):
                if os.path.exists(os.path.join(image_dir, path)):
                    shutil.copy(os.path.join(image_dir, path),
                                os.path.join(query_folder, f"top5-top{rank + 1}.jpg"))
                else:
                    print(f"Warning: top{rank + 1} image not found: {path}")

            print(f"all samples are save at: {query_folder}/")
            if len(res_list) > 10:
                break
        else:
            continue
        # save as JSON
        checkpoint_name = checkpoint_to_eval.replace('.ckpt', '')
        save_path = os.path.join(checkpoint_name + "_"+{type}+"_samples.json")

        with open(save_path, 'w') as f:
            json.dump(res_list, f, indent=4)

        print(f"Saved {len(res_list)} samples for the {type} condition.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMEBModel with PyTorch Lightning.")
    parser.add_argument('--dataset', type=str,default='cvgtext', required=True, help='Path to config YAML.')
    parser.add_argument('--subset', type=str, default='dummy', required=True, help='Path to config YAML.')
    parser.add_argument('--checkpoint_to_eval', type=str, required=True, help='Path to .ckpt file.')
    parser.add_argument('--task', type=str, default='text2image',required=True, help='task name.')
    args = parser.parse_args()
    dataset = args.dataset
    subset = args.subset

    
    if '-v' in args.checkpoint_to_eval:
        epoch = args.checkpoint_to_eval.replace('.ckpt', '').split('/')[-1].split('-v')[-1]
    if 'Epoch' in args.checkpoint_to_eval:
        epoch = args.checkpoint_to_eval.replace('.ckpt', '').split('/')[-1].split('Epoch')[-1]
    elif 'last' in args.checkpoint_to_eval:
        epoch = 'last'
    else:
        assert 'cannot find epoch' in args.checkpoint_to_eval
    eval_output_base_dir = args.checkpoint_to_eval.split(os.path.basename(args.checkpoint_to_eval))[0]
    print("eval output base dir: {}".format(eval_output_base_dir))
    if 'text2' in args.subset:
        # path = os.path.join(eval_output_base_dir, "cached_features/", args.subset + "/")
        path = os.path.join(eval_output_base_dir, "cached_features/")
    else:
        path = os.path.join(eval_output_base_dir, "cached_features/")
    qry_embeddings_path = path+'qry_embeddings'+'_Epoch'+epoch+'.npy'
    pos_embeddings_path = path+'pos_embeddings'+'_Epoch'+epoch+'.npy'
    ids_path = path+'ids'+'_Epoch'+epoch+'.npy'


    all_qry_tensor_np = np.load(qry_embeddings_path)
    all_pos_tensor_np = np.load(pos_embeddings_path)
    all_ids_np= np.load(ids_path)
    print("loading done")
    image_dir = "/gpfs2/scratch/ychen57/Datasets/GeoText/test"
    print("all_ids_np:", all_ids_np)
    if args.task=="text2image":
        scores_qry2pos_batch, _, preds_id_qry2pos_batch,preds_id_qry2pos = get_pred_batched(all_qry_tensor_np, all_pos_tensor_np,
                                                                           all_ids_np)

        # vis(scores_qry2pos_batch, args.checkpoint_to_eval, image_dir, all_ids_np, type="success")
        # vis(scores_qry2pos_batch, args.checkpoint_to_eval, image_dir, all_ids_np, type="fail")

        recall_qry2pos = compute_recall_at_k(all_ids_np, preds_id_qry2pos)
        print("recall_qry2pos: {}".format(recall_qry2pos))
        print("get recall done")
        res= recall_qry2pos
    elif args.task=="image2text":
        scores_pos2qry,_, preds_pos2qry, preds_id_pos2qry = get_pred_batched(all_pos_tensor_np, all_qry_tensor_np,
                                                                           all_ids_np)
        recall_pos2qry = compute_recall_at_k(all_ids_np, preds_id_pos2qry)
        print("recall_pos2qry: {}".format(recall_pos2qry))
        print("get recall done")
        res= recall_pos2qry


    json_output_dir = os.path.join(path)
    os.makedirs(json_output_dir, exist_ok=True)

    json_output_filename = f"res-[{dataset}]-{args.task}.json"
    json_output_path = os.path.join(json_output_dir, json_output_filename)

    with open(json_output_path, 'w') as f:
        json.dump(res, f, indent=4)
    print("results:",res)
