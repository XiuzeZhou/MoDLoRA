# MoDLoRA: Modality-Decoupled Low-Rank Adaptation for LLM-based Multimodal Explainable Recommendation
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

uiAdapter: User-Item Adapter for LLM-based Explainable Recommendation Systems

uiAdapter is an efficient fine-tuning strategy for LLM-based explainable recommendations. It decouples the adaptation processes of text and recommendation. 

Recommendation paradigms are revised from their original projects:

- **CIER**: https://github.com/karrich/CIER
- **PEPLER**: https://github.com/lileipisces/PEPLER

## 🧠uiAdapter Framework
<p align="center">
<img src="figures/uiAdapter.png" width="700">
</p>



## 📂 Project Structure

Organize the data as follows:
```
├── CIER/
|   ├── analysis_results/
|   ├── checkpoints/
│   ├── logs/
│   ├── shell/
│   ├── analysis.py
│   ├── bleu.py
│   ├── dataloader.py
│   ├── main.py
│   ├── model.py
│   ├── rouge.py
│   └── utils.py
├── PEPLER/
|   ├── analysis_results/
|   ├── checkpoints/
│   ├── logs/
│   ├── shell/
│   ├── analysis.py
│   ├── bleu.py
│   ├── main.py
│   ├── module.py
│   ├── rating_prediction.py
│   ├── rouge.py
│   └── utils.py
├── data/
│   ├── ClothingShoesAndJewelry/
│   ├── MoviesAndTV/
│   └── TripAdvisor/
├── figures/
│   └── uiAdapter.png
├── llms/
│   ├── gemma-7b/
│   ├── Mistral-7B-Instruct-v0.3/
│   └── Qwen2.5-7B/
└── requirements.txt
```

## 🛠️ Installation

Prerequisites
- Python 3.10+
- torch >= 2.7.0
- CUDA-enabled GPU (e.g., RTX 5090)

```bash
git clone https://github.com/XiuzeZhou/uiAdapter.git
cd uiAdapter
pip install -r requirements.txt
```

## 📖 Usage

### 1. Download Datasets
​    1). Download dataset from PEPLER Github: [Google Drive](https://drive.google.com/drive/folders/1yB-EFuApAOJ0RzTI0VfZ0pignytguU0_)

- ClothingShoesAndJewelry.zip
- MoviesAndTV.zip
- TripAdvisor.zip

​    2). Move .zip files to `data/` and unzip. Organize the data as follows:

```
├── data/
│   ├── ClothingShoesAndJewelry/
│   ├── MoviesAndTV/
│   └── TripAdvisor/
```


### 2. Pre-trained Models

​    1). Download the pretrained models (Gemma-7b, Mistral-7B, Qwen2.5-7B) from Hugging Face

- **Gemma-7b**: [google/gemma-7b](https://huggingface.co/google/gemma-7b)
- **Mistral-7B**: [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **Qwen2.5-7B**: [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)

​    2). Move pretrained models to  `llms/`, and organize the data as follows:

```
├── llms/
│   ├── gemma-7b/
│   ├── Mistral-7B-Instruct-v0.3/
│   └── Qwen2.5-7B/
```

### 3. Quick Start (Training)

​    1). Locate to one recommendation paradigm

```bash
cd CIER
```

or

```bash
cd PEPLER
```

​    2). To train our model with the optimal parameters:

```bash
./shell/train.sh 0
```
### 4. SVD Analysis
   To verify the modality independence and spectral distribution:
```
python analysis.py --dataset_name ClothingShoesAndJewelry
```

## 📝 Citation

If you find this work useful in your research, please consider citing:
```

```
