import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from typing import List
from scipy.stats import pearsonr
from transformers import AutoTokenizer

from module import uiAdapter, MultiModalLoraLayer
from utils import DataLoader, Batchify, now_time

# --- 1. Modality Analysis (Text vs UI/Image Collaborative) ---

def analyze_correlation(model, test_data, device, num_batches=10):
    """
    Modality Analysis (Text vs UI/Image Collaborative) for uiAdapter-based models
    """
    model.eval()
    metrics = {
        'txt_ui': {'cos': [], 'pearson': []},
        'txt_img': {'cos': [], 'pearson': []},
        'ui_img': {'cos': [], 'pearson': []}
    }

    # Cosine Correlation
    def get_cos_sim(v1, v2):
        v1 = v1.flatten().to(torch.float32)
        v2 = v2.flatten().to(torch.float32)
        return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

    # Pearson Correlation Coefficient
    def get_pearson(v1, v2):
        v1_np = v1.flatten().detach().cpu().to(torch.float32).numpy()
        v2_np = v2.flatten().detach().cpu().to(torch.float32).numpy()
        if np.std(v1_np) == 0 or np.std(v2_np) == 0:
            return 0.0
        corr, _ = pearsonr(v1_np, v2_np)
        return corr if not np.isnan(corr) else 0.0

    print("\n" + "="*60)
    print("Start Modality Analysis (Text vs UI/Image Collaborative)...")

    test_data.step = 0
    pad_id = test_data.tokenizer.pad_token_id if hasattr(test_data, 'tokenizer') else None

    with torch.no_grad():
        for batch_idx in range(num_batches):
            user, item, rating, input_ids, mask, prompt_ids, prompt_text, prompt_lens, review_text, text_lens = test_data.next_batch()
            
            current_batch_size = input_ids.size(0)
            user = user.to(device)
            item = item.to(device)
            input_ids = input_ids.to(device)

            x_ui_computed, x_img_computed = model.build_lora_context(user, item)
            model.set_Q(x_ui_computed, x_img_computed)
            
            target_layer = None
            for m in model.modules():
                if isinstance(m, MultiModalLoraLayer) and m.x_ui is not None:
                    target_layer = m
                    break
            
            if target_layer is None:
                continue

            # 3. Extract multi-modal features
            embeddings_layer = model.get_input_embeddings()
            token_embeddings = embeddings_layer(input_ids).to(torch.float32) # [Batch, Seq_len, Dim]
            
            if pad_id is not None:
                txt_mask = (input_ids != pad_id).unsqueeze(-1).to(torch.float32) # [Batch, Seq_len, 1]
                x_txt = (token_embeddings * txt_mask).sum(dim=1) / txt_mask.sum(dim=1).clamp(min=1)
            else:
                x_txt = token_embeddings.mean(dim=1)
            
            x_ui = target_layer.x_ui[:current_batch_size].to(torch.float32)
            x_img = target_layer.x_img[:current_batch_size].to(torch.float32)

            for b in range(current_batch_size):
                metrics['txt_ui']['cos'].append(get_cos_sim(x_txt[b], x_ui[b]))
                metrics['txt_ui']['pearson'].append(get_pearson(x_txt[b], x_ui[b]))
                metrics['txt_img']['cos'].append(get_cos_sim(x_txt[b], x_img[b]))
                metrics['txt_img']['pearson'].append(get_pearson(x_txt[b], x_img[b]))
                metrics['ui_img']['cos'].append(get_cos_sim(x_ui[b], x_img[b]))
                metrics['ui_img']['pearson'].append(get_pearson(x_ui[b], x_img[b]))

            if test_data.step >= test_data.total_step:
                print("The end of the test dataset has been reached, and the data iteration is exited early.")
                break

    print(f"{'Modal Pair':<15} | {'Cosine Sim':<15} | {'Pearson Corr':<15}")
    print("-" * 50)
    for key, val in metrics.items():
        if val['cos']:
            avg_cos = np.mean(val['cos'])
            avg_pea = np.mean(val['pearson'])
            print(f"{key:<15} | {avg_cos:^15.4f} | {avg_pea:^15.4f}")
    print("="*60)

    return metrics


# --- 2. SVD Analysis ---

def compute_delta_w(lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    """Delta W = B @ A * scaling"""
    lora_A = lora_A.to(torch.float32)
    lora_B = lora_B.to(torch.float32)
    return (lora_B @ lora_A) * scaling


def analyze_svd_of_lora_weights(model: uiAdapter, num_singular_values: int = 100, lora_id: int = 0, output_dir: str = './analysis_results'):
    """
    SVD Analysis for uiAdapter-based Weights
    
    1. 'Text_$\Delta$W', 'UI_$\Delta$W', and 'Image_$\Delta$W'
    2. 'Fused_(Text+UI+Image)_$\Delta$W'
    """
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "="*80)
    print("Start SVD Analysis for uiAdapter-based Model ...")
    
    results = {}
    
    lora_layers: List[tuple[str, MultiModalLoraLayer]] = []
    for name, module in model.named_modules():
        if isinstance(module, MultiModalLoraLayer):
            lora_layers.append((name, module))

    if not lora_layers:
        print("Cannot find MultiModalLoraLayer in models!")
        return

    # Select MultiModalLoraLayer for analysis
    layer_name, lora_layer = lora_layers[lora_id]
    scaling = lora_layer.scaling
    ui_scaling = lora_layer.ui_multimodal_scaling.detach()
    image_scaling = lora_layer.image_multimodal_scaling.detach()
    
    # Adapter rank
    r_val = lora_layer.lora_A_t.shape[0]
    
    print(f"Selected Layer: {layer_name}")
    print(f"Adapter Rank (r): {r_val}")
    print(f"UI scaling: {ui_scaling.item():.4f} | Image scaling: {image_scaling.item():.4f}")

    # 1. Textual Branch: Delta W_t
    delta_w_t = compute_delta_w(lora_layer.lora_A_t.data, lora_layer.lora_B_t.data, scaling)
    
    # 2. UI Branch: single Delta W_ui
    delta_w_ui = compute_delta_w(lora_layer.lora_A_ui.data, lora_layer.lora_B_ui.data, scaling * ui_scaling)

    # 3. Image Branch: single Delta W_img
    delta_w_img = compute_delta_w(lora_layer.lora_A_img.data, lora_layer.lora_B_img.data, scaling * image_scaling)

    delta_w_fused = delta_w_t + delta_w_ui + delta_w_img

    group1_matrices = {
        'Text_$\\Delta$W': delta_w_t,
        'UI_$\\Delta$W': delta_w_ui,
        'Image_$\\Delta$W': delta_w_img
    }
    
    group2_matrices = {
        'Fused_(Text+UI+Image)_$\\Delta$W': delta_w_fused
    }

    rank_checkpoints_group1 = [r_val]
    labels_chk_group1 = [f'$r={r_val}$']
    rank_checkpoints_group2 = [r_val * 3]
    labels_chk_group2 = [f'$3r={r_val * 3}$']
    colors = ['blue', 'red', 'green']

    def plot_matrix_group(matrices_dict, filename_suffix, title_suffix, rank_checkpoints, labels_chk):
        plt.figure(figsize=(10, 6))
        
        for label, delta_w in matrices_dict.items():
            delta_w_f32 = delta_w.to(torch.float32)
            s = torch.linalg.svdvals(delta_w_f32).cpu().numpy()
            
            num_to_plot = min(len(s), num_singular_values)
            s_plot = s[:num_to_plot]
            indices = np.arange(1, num_to_plot + 1)
            
            variance_ratio = np.sum(s_plot**2) / np.sum(s**2) if np.sum(s**2) > 0 else 0
            
            results[label] = {
                'total_rank': delta_w.shape[0],
                f'top_{num_to_plot}_variance_ratio': variance_ratio
            }
            
            print(f"\n-> SVD Results ({label})")
            print(f"    Shape: {delta_w.shape[0]}x{delta_w.shape[1]}")
            print(f"    First {num_to_plot} singular value ratio: {variance_ratio * 100:.2f}%")
            
            plt.plot(indices, s_plot, marker='.', linestyle='-', markersize=4, label=label)

        plt.xscale('log')
        plt.yscale('log')
        #plt.title(f'Singular Value Spectrum ({title_suffix}) in {layer_name.split(".")[-1]}', fontsize=12)
        plt.xlabel('Singular Value Index $\\log(k)$', fontsize=10)
        plt.ylabel('Singular Value $\\log(\\sigma_k)$', fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='best')
        
        ymin, ymax = plt.ylim()
        if ymax <= 0:
            ymax = 1.0
            
        for rc, lbl, col in zip(rank_checkpoints, labels_chk, colors):
            plt.axvline(x=rc, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
            plt.text(rc * 1.05, ymax * 0.2, lbl, color=col, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename_suffix)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Figures saved to: {plot_path}")

    # Figure One: Text_ΔW 和 UI_ΔW
    plot_matrix_group(group1_matrices, 'svd_spectrum_text_ui_image.png', 'Text & UI & Image Branches', rank_checkpoints_group1, labels_chk_group1)

    # Figure Two: Fused_(Text+UI)_ΔW
    plot_matrix_group(group2_matrices, 'svd_spectrum_fused.png', 'Fused (Text+UI+Image)', rank_checkpoints_group2, labels_chk_group2)

    print(f"\n" + "="*80)
    print(f"SVD figures saved to: {output_dir}")
    print("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='User-Item Adapter for LLM-based Explainable Recommendation Systems (uiAdapter)')
    parser.add_argument('-dataset_name', type=str, default='ClothingShoesAndJewelry')
    parser.add_argument('-data_path', '--data_path', type=str, default='./data/ClothingShoesAndJewelry/reviews.pickle')
    parser.add_argument('-index_dir', '--index_dir', type=str, default='./data/ClothingShoesAndJewelry/1/')
    parser.add_argument('-llm_model', '--llm_model', type=str, default="../llms/Qwen2.5-7B/")
    parser.add_argument('-clip_model', '--clip_model', type=str, default="./llm/clip-vit-base-patch32/")
    parser.add_argument('-checkpoint', '--checkpoint', type=str, default='./checkpoints/')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=16)
    parser.add_argument('-words', '--words', type=int, default=20)
    parser.add_argument('-mlp_size', '--mlp_size', type=int, default=400)
    parser.add_argument('-k', '--k', type=int, default=768)
    parser.add_argument('-r', '--r', type=int, default=8)
    parser.add_argument('-expert_number', '--expert_number', type=int, default=4)
    parser.add_argument('-top_k', '--top_k', type=int, default=2)
    parser.add_argument('-lora_modules', '--lora_modules', type=int, default=7)
    parser.add_argument('-image_dir', '--image_dir', type=str, default='images')
    parser.add_argument('-ui_multimodal_scale', '--ui_multimodal_scale', type=float, default=1.0)
    parser.add_argument('-image_multimodal_scale', '--image_multimodal_scale', type=float, default=1.0)
    parser.add_argument('-cuda', '--cuda', action='store_true', default=True)
    parser.add_argument('-num_batches', type=int, default=10, help='batch size')
    parser.add_argument('-output_dir', type=str, default='./analysis_results')
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # 1. Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side='left', spaces_between_special_tokens=False) 
    bos = tokenizer.bos_token or '<s>'
    eos = tokenizer.eos_token or '</s>'
    pad = tokenizer.pad_token or tokenizer.eos_token
    if 'gpt2' in args.llm_model.lower() or 'gpt-2' in args.llm_model.lower():
        pad = tokenizer.pad_token or '<|endoftext|>'
    tokenizer.pad_token = pad
    tokenizer.add_special_tokens({'bos_token': bos, 'eos_token': eos, 'pad_token': pad})
    
    corpus = DataLoader(
        args.data_path, args.index_dir, tokenizer, args.words,
        clip_path=args.clip_model,
        image_dir=args.image_dir,
        device=device
    )
    test_data = Batchify(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, 
                          args.words, args.batch_size, corpus.max_rating, corpus.min_rating)

    # 2. load the pretrained weights
    model = uiAdapter.from_pretrained(
        args.llm_model, 
        len(corpus.user_dict), 
        len(corpus.item_dict), 
        args.k, 
        args.r, 
        args.mlp_size, 
        args.lora_modules, 
        args.expert_number, 
        args.top_k,
        image_embeddings=corpus.image_embeddings,
        ui_multimodal_scale=args.ui_multimodal_scale,
        image_multimodal_scale=args.image_multimodal_scale
    )
    model.resize_token_embeddings(len(tokenizer))
    
    model_path = os.path.join(args.checkpoint, args.dataset_name, 'model.pt')
    if os.path.exists(model_path):
        print(now_time() + f"load the pretrained weights from: {model_path}")
        loaded_state = torch.load(model_path, map_location=device)
        model.load_state_dict(loaded_state, strict=False)
        print(now_time() + f'Successfully loaded trainable parameters from {model_path}')
        model = model.to(device)
    else:
        print(now_time() + f"WARNING: Can not find checkpoint: {model_path}")
        
    model.to(device)

    # 3. Analysis of Subspace Decoupling and Spectral Extension
    lora_id = 0
    analyze_correlation(model, test_data, device, num_batches=args.num_batches)
    analyze_svd_of_lora_weights(model, num_singular_values=64, lora_id=lora_id, output_dir=args.output_dir)
