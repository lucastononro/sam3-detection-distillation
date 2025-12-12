# SAM 3 Detection & Distillation

Examples and experiments with Meta's [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3).

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sam3-detection-distillation.git
   cd sam3-detection-distillation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up HuggingFace authentication**

   SAM 3 weights are gated. You need to:
   - Request access at https://huggingface.co/facebook/sam3
   - Create a token at https://huggingface.co/settings/tokens (enable "Access to public gated repositories")
   - Copy `.env.example` to `.env` and add your token:
     ```bash
     cp .env.example .env
     # Edit .env with your token
     ```

## Examples

### Jupyter Notebook

```bash
jupyter notebook examples/notebooks/sam3_demo.ipynb
```

### Streamlit App

```bash
streamlit run examples/streamlit-app/app.py
```

The app allows you to:
- Upload images (PNG, JPEG, WebP, BMP, GIF, TIFF)
- Enter text prompts to segment objects
- Adjust detection threshold
- View segmentation masks with confidence scores

## Project Structure

```
sam3-detection-distillation/
├── examples/
│   ├── notebooks/
│   │   └── sam3_demo.ipynb      # Jupyter notebook demo
│   └── streamlit-app/
│       └── app.py               # Streamlit web app
├── .env.example                 # Environment template
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA, MPS (Apple Silicon), or CPU

## Resources

- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM 3 on HuggingFace](https://huggingface.co/facebook/sam3)
- [SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Meta AI Blog Post](https://ai.meta.com/blog/segment-anything-model-3/)
