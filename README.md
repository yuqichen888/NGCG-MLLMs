# Turning Generators into Retrievers: Unlocking MLLMs for Natural Language-Guided Geo-Localization

<p align="center">
  <a href="https://yuqichen888.github.io/NGCG-MLLMs-web/">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Website"/>
  </a>
  <a href="#-citation">
    <img src="https://img.shields.io/badge/cite-BibTeX-orange?style=flat-square" alt="BibTeX"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.10-green?style=flat-square&logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="License"/>
</p>

Official implementation of **"Turning Generators into Retrievers: Unlocking MLLMs for Natural Language-Guided Geo-Localization"** (CVPR 2026, Findings).

This repository provides a framework for leveraging Multimodal Large Language Models (MLLMs) for precise, natural language-guided geo-localization by transforming generative capabilities into retrieval strengths.

---

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Data Preparation](#-data-preparation)
- [Usage Guide](#-usage-guide)
- [Configuration & Data Fields](#️-configuration--data-fields)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 📂 Repository Structure

```
.
├── train.py              # Main training for model training and fine-tuning
├── eva.py                # Evaluation for CVGText and GeoText
├── eva_compute.py        # (GeoText only) Large-scale metric calculator
├── src/                  # Core source: model definitions, losses, data utilities
└── config/               # YAML configs for each model architecture
    ├── internvl.yaml
    └── smolvlm.yaml
```

> **`eva_compute.py` is required for GeoText evaluation.** Running metric computation inside `eva.py` on the full GeoText split will cause Out-Of-Memory (OOM) errors on standard GPUs.

---

## 🛠 Setup & Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- CUDA-compatible GPU

### 1. Create Environment

```bash
conda create -n ngcg python=3.10
conda activate ngcg
pip install -r requirements.txt
```

---

## 📁 Data Preparation

Organize your data directory as follows:

```plaintext
data/
├── CVGText/
│   ├── images/
│   │   └── <city>/             
│   ├── texts/
│   │   └── <city>/
│   ├── text2sate_<city>_train.json
│   └── text2sate_<city>_test.json
│   ├── text2osm_<city>_train.json
│   └── text2osm_<city>_test.json
└── GeoText/
    ├── images/
    │   ├── train/
    │   └── test/  
    ├── geo_t2i_train.json
    └── geo_t2i_test.json
```

All JSON files follow the original dataset author's format. We augment each entry with four additional fields to unify the retrieval pipeline — see [Configuration & Data Fields](#️-configuration--data-fields) for details.

---

## 🚀 Usage Guide

### 1. Training

Submit a training job on a SLURM-managed cluster:

```bash
sbatch train_internvl_geo.sh
```

### 2. Evaluation

Evaluation is handled differently depending on dataset scale.

---

#### A. CVGText (Standard)

`eva.py` runs inference and computes metrics in a single pass:

```bash
python eva.py \
    --config_path        <path/to/config.yaml> \
    --checkpoint_to_eval <path/to/checkpoint> \
    --dataset_name       CVGText \
    --subset_name        vec_text2sate_newyork \
    --augmentation       'none'
```

---

#### B. GeoText (Large-Scale, Two-Stage)

Due to the scale of GeoText, evaluation is split into two stages.

**Stage 1 — Extract Features**

```bash
python eva.py \
    --config_path        <path/to/config.yaml> \
    --checkpoint_to_eval <path/to/checkpoint> \
    --dataset_name       GeoText \
    --task               text2image \
    --subset_name        geo_24G_t2i \
    --augmentation       'none'
```

**Stage 2 — Compute Recall@K Metrics**

```bash
python eva_compute.py \
    --checkpoint_to_eval <path/to/checkpoint> \
    --dataset            GeoText \
    --subset             geo_t2i \
    --task               image2text
```

---

## ⚙️ Configuration & Data Fields

We augment JSON annotation files with four additional fields to support the our framework.  If the query is text-to-image, you would feed the text into qry_text_field and the corresponding ground-truth image into pos_img_path_field. 

| Field | Description |
| :--- | :--- |
| `qry_text_field` | Natural language query description. |
| `qry_img_path_field` | Path to the query image (if applicable). |
| `pos_text_field` | Text field of the positive (ground-truth) pair. |
| `pos_img_path_field` | Path to the ground-truth target image. |

These fields are consistent across both CVGText and GeoText and are required for the data loader to function correctly.

---
## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{chen2026turning,
  title     = {Turning Generators into Retrievers: Unlocking MLLMs for Natural Language-Guided Geo-Localization},
  author    = {Yuqi Chen and Xiaohan Zhang and Ahmad Arrabi and Waqas Sultani and Chen Chen and Safwan Wshah},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year      = {2026}
}
```

---

## 📧 Contact

For questions about the paper or code, please open a GitHub issue or contact **Yuqi Chen**.
