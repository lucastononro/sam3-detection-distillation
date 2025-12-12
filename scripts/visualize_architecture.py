"""
SAM3 Architecture Visualization Script

This script loads the SAM3 model and visualizes its architecture,
showing layers, parameter counts, and model structure.
"""

import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
import torch

# Load environment variables
load_dotenv()

def setup_auth():
    """Setup HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("✓ Authenticated with HuggingFace")
    else:
        print("⚠ No HF_TOKEN found in .env file")
        sys.exit(1)

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_params(num):
    """Format parameter count in human readable form."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)

def print_model_summary(model, model_name="SAM3"):
    """Print a summary of the model architecture."""
    total, trainable = count_parameters(model)

    print("\n" + "="*70)
    print(f" {model_name} Architecture Summary")
    print("="*70)
    print(f"\nTotal Parameters:     {format_params(total)} ({total:,})")
    print(f"Trainable Parameters: {format_params(trainable)} ({trainable:,})")
    print(f"Non-trainable:        {format_params(total-trainable)} ({total-trainable:,})")
    print("="*70)

def print_layer_breakdown(model):
    """Print detailed breakdown of each major component."""
    print("\n" + "-"*70)
    print(" Layer Breakdown")
    print("-"*70)

    components = {}

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        components[name] = params

    # Sort by parameter count
    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Component':<40} {'Parameters':>15} {'%':>10}")
    print("-"*70)

    total = sum(components.values())
    for name, params in sorted_components:
        pct = (params / total * 100) if total > 0 else 0
        print(f"{name:<40} {format_params(params):>15} {pct:>9.1f}%")

    print("-"*70)
    print(f"{'TOTAL':<40} {format_params(total):>15} {'100.0%':>10}")

def print_detailed_architecture(model, max_depth=3):
    """Print detailed architecture with nested modules."""
    print("\n" + "-"*70)
    print(" Detailed Architecture")
    print("-"*70 + "\n")

    def print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return

        for name, child in module.named_children():
            params = sum(p.numel() for p in child.parameters())
            child_count = len(list(child.children()))

            # Get module type
            module_type = child.__class__.__name__

            if params > 0:
                param_str = f"[{format_params(params)}]"
            else:
                param_str = ""

            print(f"{prefix}├── {name}: {module_type} {param_str}")

            if child_count > 0 and depth < max_depth:
                print_module(child, prefix + "│   ", depth + 1)

    print_module(model)

def print_config_info(model):
    """Print model configuration information."""
    if hasattr(model, 'config'):
        config = model.config
        print("\n" + "-"*70)
        print(" Model Configuration")
        print("-"*70 + "\n")

        # Print key config values
        config_dict = config.to_dict()
        important_keys = [
            'model_type', 'hidden_size', 'num_attention_heads',
            'num_hidden_layers', 'intermediate_size', 'image_size',
            'patch_size', 'num_channels'
        ]

        for key in important_keys:
            if key in config_dict:
                print(f"  {key}: {config_dict[key]}")

        # Check nested configs
        if hasattr(config, 'vision_config') and config.vision_config:
            print("\n  Vision Config:")
            if hasattr(config.vision_config, 'to_dict'):
                for k, v in config.vision_config.to_dict().items():
                    if not k.startswith('_') and v is not None:
                        print(f"    {k}: {v}")

def visualize_with_torchview(model, device):
    """Create a visual graph of the model using torchview."""
    try:
        from torchview import draw_graph
        from PIL import Image

        print("\n" + "-"*70)
        print(" Generating Visual Graph...")
        print("-"*70)

        # Create dummy inputs
        batch_size = 1
        image_size = 1008  # SAM3 default

        dummy_pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device)
        dummy_input_ids = torch.randint(0, 1000, (batch_size, 10)).to(device)
        dummy_attention_mask = torch.ones(batch_size, 10).to(device)

        # Generate graph
        model_graph = draw_graph(
            model,
            input_data={
                'pixel_values': dummy_pixel_values,
                'input_ids': dummy_input_ids,
                'attention_mask': dummy_attention_mask,
            },
            expand_nested=True,
            depth=4,
            save_graph=True,
            filename="outputs/sam3_architecture",
            directory="."
        )

        print("✓ Graph saved to: outputs/sam3_architecture.png")
        return True

    except Exception as e:
        print(f"⚠ Could not generate visual graph: {e}")
        return False

def export_architecture_to_file(model, filepath="outputs/sam3_architecture.txt"):
    """Export full architecture to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("SAM3 Full Architecture\n")
        f.write("="*70 + "\n\n")
        f.write(str(model))

    print(f"\n✓ Full architecture exported to: {filepath}")

def main():
    print("\n" + "="*70)
    print(" SAM3 Architecture Visualization")
    print("="*70)

    # Setup
    setup_auth()
    device = get_device()
    print(f"✓ Using device: {device}")

    # Load model
    print("\nLoading SAM3 model...")
    from transformers.models.sam3 import Sam3Model

    model = Sam3Model.from_pretrained("facebook/sam3")
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")

    # Print summaries
    print_model_summary(model)
    print_config_info(model)
    print_layer_breakdown(model)
    print_detailed_architecture(model, max_depth=2)

    # Export full architecture
    os.makedirs("outputs", exist_ok=True)
    export_architecture_to_file(model)

    # Try visual graph (may fail on MPS/CPU)
    if device == "cuda":
        visualize_with_torchview(model, device)
    else:
        print(f"\n⚠ Visual graph generation skipped (requires CUDA, got {device})")

    print("\n" + "="*70)
    print(" Done!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
