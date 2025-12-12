import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="SAM 3 Segmentation Demo",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load SAM3 model (cached to avoid reloading)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    from transformers.models.sam3 import Sam3Processor, Sam3Model

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    return model, processor, device

def segment_image(image, text_prompt, model, processor, device, threshold=0.5):
    """Run SAM3 segmentation on an image."""
    # Process inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=[image.size[::-1]]
    )[0]

    return results

def create_overlay(image, masks, boxes, scores, alpha=0.5):
    """Create visualization with masks overlaid on image."""
    img_array = np.array(image).copy()

    # Generate random colors for each mask
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(masks), 3))

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        # Convert mask to numpy
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Create colored overlay
        color = colors[i]
        mask_bool = mask_np.astype(bool)

        # Apply color overlay
        for c in range(3):
            img_array[:, :, c] = np.where(
                mask_bool,
                img_array[:, :, c] * (1 - alpha) + color[c] * alpha,
                img_array[:, :, c]
            )

        # Draw bounding box
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Draw box border
        thickness = 3
        img_array[y1:y1+thickness, x1:x2] = color
        img_array[y2-thickness:y2, x1:x2] = color
        img_array[y1:y2, x1:x1+thickness] = color
        img_array[y1:y2, x2-thickness:x2] = color

    return Image.fromarray(img_array.astype(np.uint8))

# Main app
st.title("🎯 SAM 3 Segmentation Demo")
st.markdown("Upload an image and enter a text prompt to segment objects using Meta's SAM 3 model.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    overlay_alpha = st.slider("Overlay Transparency", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **SAM 3** (Segment Anything Model 3) is Meta's latest
    foundation model for image segmentation with text prompts.

    [GitHub](https://github.com/facebookresearch/sam3) |
    [Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
    """)

# Load model
with st.spinner("Loading SAM 3 model... (this may take a minute on first run)"):
    try:
        model, processor, device = load_model()
        st.sidebar.success(f"Model loaded on: {device}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("Input")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"]
    )

    # Text prompt
    text_prompt = st.text_input(
        "Enter text prompt",
        placeholder="e.g., dog, car, person, building..."
    )

    # Segment button
    segment_button = st.button("🔍 Segment", type="primary", use_container_width=True)

with col2:
    st.header("Output")
    output_placeholder = st.empty()

# Process image
if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if segment_button and text_prompt:
        with st.spinner(f"Segmenting '{text_prompt}'..."):
            try:
                # Run segmentation
                results = segment_image(
                    image, text_prompt, model, processor, device, threshold
                )

                masks = results["masks"]
                boxes = results["boxes"]
                scores = results["scores"]

                if len(masks) > 0:
                    # Create overlay
                    overlay = create_overlay(image, masks, boxes, scores, overlay_alpha)

                    with col2:
                        output_placeholder.image(overlay, caption=f"Found {len(masks)} '{text_prompt}'", use_container_width=True)

                        # Show details
                        st.markdown("### Detection Details")
                        for i, score in enumerate(scores):
                            score_val = score.item() if hasattr(score, 'item') else score
                            st.write(f"**Instance {i+1}:** confidence = {score_val:.2%}")
                else:
                    with col2:
                        output_placeholder.warning(f"No '{text_prompt}' found in the image. Try a different prompt or lower the threshold.")

            except Exception as e:
                st.error(f"Error during segmentation: {e}")

    elif segment_button and not text_prompt:
        st.warning("Please enter a text prompt.")

else:
    with col1:
        st.info("👆 Upload an image to get started")
    with col2:
        output_placeholder.info("Results will appear here")

# Example prompts
st.markdown("---")
st.markdown("### Example Prompts")
example_cols = st.columns(6)
examples = ["dog", "cat", "person", "car", "building", "tree"]
for col, example in zip(example_cols, examples):
    col.code(example)
