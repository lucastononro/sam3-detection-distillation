"""
SAM3 Embedding Similarity Script

Extract vision and text embeddings from SAM3 and compute dot product similarity.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path

# Load environment variables
load_dotenv()


def setup():
    """Setup authentication and device."""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("✓ Authenticated with HuggingFace")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"✓ Using device: {device}")
    return device


def load_model(device):
    """Load SAM3 model."""
    print("\nLoading SAM3 model...")
    from transformers.models.sam3 import Sam3Processor, Sam3Model

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    print("✓ Model loaded!")
    return model, processor


def get_vision_embeddings(model, processor, image, device):
    """Extract vision embeddings from an image."""
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get vision features using the model's method
        vision_outputs = model.get_vision_features(pixel_values=inputs.pixel_values)

    return vision_outputs, inputs


def get_text_embeddings(model, processor, text_prompts, device):
    """Extract text embeddings from text prompts."""
    # Process text
    text_inputs = processor(text=text_prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get text features
        text_outputs = model.get_text_features(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )

    return text_outputs, text_inputs


def compute_similarity_via_model(model, processor, image, text_prompt, device):
    """
    Compute similarity using the model's full forward pass.
    Returns the presence score and detection scores.
    """
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get scores from the model output
    # pred_logits: confidence scores for each query
    # presence_logits: whether the concept is present at all
    pred_logits = outputs.pred_logits  # [batch, num_queries]
    presence_logits = outputs.presence_logits  # [batch, 1]

    # Compute final scores (as SAM3 does internally)
    pred_scores = torch.sigmoid(pred_logits)
    presence_score = torch.sigmoid(presence_logits)

    # Combined score
    final_scores = pred_scores * presence_score

    return {
        "presence_score": presence_score.item(),
        "max_detection_score": pred_scores.max().item(),
        "mean_detection_score": pred_scores.mean().item(),
        "final_max_score": final_scores.max().item(),
        "num_detections": (final_scores > 0.5).sum().item(),
    }


def compute_embedding_similarity(vision_embeds, text_embeds, model=None, method="cosine"):
    """
    Compute similarity between vision and text embeddings directly.
    Uses the model's dot_product_scoring module if available.
    """
    # Get the pooled/mean representation
    if len(text_embeds.shape) == 3:
        text_pooled = text_embeds.mean(dim=1)
    else:
        text_pooled = text_embeds

    # For vision, handle the feature pyramid
    if hasattr(vision_embeds, 'last_hidden_state'):
        vision_feat = vision_embeds.last_hidden_state
    elif isinstance(vision_embeds, tuple):
        vision_feat = vision_embeds[0]
    else:
        vision_feat = vision_embeds

    # Spatial average pooling if needed
    if len(vision_feat.shape) == 4:
        vision_pooled = vision_feat.mean(dim=[2, 3])
    elif len(vision_feat.shape) == 3:
        vision_pooled = vision_feat.mean(dim=1)
    else:
        vision_pooled = vision_feat

    # Use model's projection layers if available
    if model is not None and hasattr(model, 'dot_product_scoring'):
        # Use the model's scoring module
        scoring = model.dot_product_scoring

        # Project text through the text MLP
        text_projected = scoring.text_mlp(text_pooled)
        text_projected = scoring.text_mlp_dropout(text_projected)
        text_projected = scoring.text_mlp_out_norm(text_projected)
        text_projected = scoring.text_proj(text_projected)

        # For vision, we need query features (from decoder)
        # Since we only have vision encoder features, project them
        vision_projected = scoring.query_proj(
            torch.nn.functional.adaptive_avg_pool1d(
                vision_pooled.unsqueeze(0), text_projected.shape[-1]
            ).squeeze(0)
        ) if vision_pooled.shape[-1] != text_projected.shape[-1] else scoring.query_proj(vision_pooled)

        vision_norm = torch.nn.functional.normalize(vision_projected, dim=-1)
        text_norm = torch.nn.functional.normalize(text_projected, dim=-1)
        similarity = torch.matmul(vision_norm, text_norm.T)
    else:
        # Fallback: project to common dimension
        vision_dim = vision_pooled.shape[-1]
        text_dim = text_pooled.shape[-1]
        common_dim = min(vision_dim, text_dim)

        # Simple linear projections
        vision_proj = torch.nn.Linear(vision_dim, common_dim, bias=False).to(vision_pooled.device)
        text_proj = torch.nn.Linear(text_dim, common_dim, bias=False).to(text_pooled.device)
        torch.nn.init.xavier_uniform_(vision_proj.weight)
        torch.nn.init.xavier_uniform_(text_proj.weight)

        vision_projected = vision_proj(vision_pooled)
        text_projected = text_proj(text_pooled)

        vision_norm = torch.nn.functional.normalize(vision_projected, dim=-1)
        text_norm = torch.nn.functional.normalize(text_projected, dim=-1)
        similarity = torch.matmul(vision_norm, text_norm.T)

    return similarity, vision_pooled, text_pooled


def compute_spatial_similarity(model, vision_embeds, text_embeds, device):
    """
    Compute per-pixel similarity map between vision features and text.
    This shows WHERE in the image the text concept is located.
    """
    # Get vision features at different scales
    if hasattr(vision_embeds, 'last_hidden_state'):
        vision_feat = vision_embeds.last_hidden_state
    elif isinstance(vision_embeds, tuple):
        vision_feat = vision_embeds[0]
    else:
        vision_feat = vision_embeds

    # Get text pooled representation
    if len(text_embeds.shape) == 3:
        text_pooled = text_embeds.mean(dim=1)  # [batch, hidden]
    else:
        text_pooled = text_embeds

    # If vision is [B, C, H, W], reshape to [B, H*W, C]
    if len(vision_feat.shape) == 4:
        B, C, H, W = vision_feat.shape
        vision_flat = vision_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    else:
        vision_flat = vision_feat
        H = W = int(np.sqrt(vision_feat.shape[1]))

    # Normalize for cosine similarity
    vision_norm = torch.nn.functional.normalize(vision_flat, dim=-1)
    text_norm = torch.nn.functional.normalize(text_pooled, dim=-1)

    # Compute similarity: [B, H*W, hidden] x [B, hidden, 1] -> [B, H*W, 1]
    similarity_map = torch.matmul(vision_norm, text_norm.unsqueeze(-1)).squeeze(-1)

    # Reshape to spatial map
    similarity_map = similarity_map.reshape(B, H, W)

    return similarity_map


def visualize_similarity_map(image, similarity_map, text_prompt, save_path=None):
    """Visualize the spatial similarity map overlaid on the image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Similarity heatmap
    sim_np = similarity_map.cpu().numpy()[0]
    im = axes[1].imshow(sim_np, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f"Similarity Map: '{text_prompt}'")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(image)
    # Resize similarity map to image size
    sim_resized = np.array(Image.fromarray(sim_np).resize(image.size, Image.BILINEAR))
    # Normalize for visualization
    sim_resized = (sim_resized - sim_resized.min()) / (sim_resized.max() - sim_resized.min() + 1e-8)
    axes[2].imshow(sim_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Overlay: '{text_prompt}'")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def compare_multiple_prompts(image, model, processor, text_prompts, device, save_path=None):
    """Compare similarity scores for multiple text prompts using model's scoring."""
    results = []
    for prompt in text_prompts:
        scores = compute_similarity_via_model(model, processor, image, prompt, device)
        results.append({
            "prompt": prompt,
            "similarity": scores["presence_score"],
            "max_score": scores["final_max_score"],
            "num_detections": scores["num_detections"],
        })

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Print results
    print("\n" + "="*60)
    print(" Text-Image Similarity Scores (Presence Score)")
    print("="*60)
    print(f"{'Prompt':<15} {'Presence':>10} {'Max Score':>10} {'Detections':>12}")
    print("-"*60)
    for r in results:
        bar = "█" * int(r["similarity"] * 20)
        print(f"{r['prompt']:<15} {r['similarity']:>10.4f} {r['max_score']:>10.4f} {r['num_detections']:>12} {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    prompts = [r["prompt"] for r in results]
    scores = [r["similarity"] for r in results]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores)))
    bars = ax.barh(prompts, scores, color=colors)

    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Text-Image Similarity Comparison")
    ax.set_xlim(0, max(scores) * 1.1)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()

    return results


def main():
    print("\n" + "="*60)
    print(" SAM3 Embedding Similarity Analysis")
    print("="*60)

    # Setup
    device = setup()
    model, processor = load_model(device)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load or download a test image
    image_path = Path("images/dog.jpg")
    if not image_path.exists():
        print("\nDownloading test image...")
        import requests
        Path("images").mkdir(exist_ok=True)
        url = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"✓ Downloaded: {image_path}")

    image = Image.open(image_path).convert("RGB")
    print(f"\n✓ Loaded image: {image_path} ({image.size})")

    # Example 1: Compare multiple prompts
    print("\n" + "-"*60)
    print(" Example 1: Compare Multiple Text Prompts")
    print("-"*60)

    prompts = ["dog", "cat", "animal", "pet", "car", "building", "tree", "grass", "sky"]
    compare_multiple_prompts(
        image, model, processor, prompts, device,
        save_path="outputs/similarity_comparison.png"
    )

    # Example 2: Spatial similarity maps for different prompts
    print("\n" + "-"*60)
    print(" Example 2: Spatial Similarity Maps")
    print("-"*60)

    for prompt in ["dog", "grass", "sky"]:
        print(f"\nComputing spatial similarity for: '{prompt}'")
        vision_embeds, _ = get_vision_embeddings(model, processor, image, device)
        text_embeds, _ = get_text_embeddings(model, processor, prompt, device)

        try:
            sim_map = compute_spatial_similarity(model, vision_embeds, text_embeds, device)
            visualize_similarity_map(
                image, sim_map, prompt,
                save_path=f"outputs/similarity_map_{prompt}.png"
            )
        except Exception as e:
            print(f"  ⚠ Could not compute spatial map: {e}")

    # Example 3: Raw embedding inspection
    print("\n" + "-"*60)
    print(" Example 3: Embedding Statistics")
    print("-"*60)

    vision_embeds, _ = get_vision_embeddings(model, processor, image, device)
    text_embeds, _ = get_text_embeddings(model, processor, "dog", device)

    print("\nVision Embeddings:")
    if hasattr(vision_embeds, 'last_hidden_state'):
        v = vision_embeds.last_hidden_state
        print(f"  Shape: {v.shape}")
        print(f"  Mean: {v.mean().item():.4f}")
        print(f"  Std: {v.std().item():.4f}")
        print(f"  Min: {v.min().item():.4f}")
        print(f"  Max: {v.max().item():.4f}")
    elif isinstance(vision_embeds, tuple):
        for i, v in enumerate(vision_embeds):
            print(f"  Scale {i}: {v.shape}")

    print("\nText Embeddings:")
    print(f"  Shape: {text_embeds.shape}")
    print(f"  Mean: {text_embeds.mean().item():.4f}")
    print(f"  Std: {text_embeds.std().item():.4f}")

    print("\n" + "="*60)
    print(" Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
