# Multimodal Food AI: Cross-Modal Recipe Intelligence

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?style=flat-square&logo=openai&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A cross-modal food intelligence system that bridges food images and recipe text through joint embedding learning. Given a food photograph, the system retrieves semantically similar recipes from a corpus of 2.2M. Given a recipe, it finds visually matching food images. Built on the Recipe1M image dataset and RecipeNLG text corpus using CLIP, EfficientNet, and sentence-transformers.

**Live demo:** [huggingface.co/spaces/drnsmith/multimodal-food-ai](https://huggingface.co/spaces/drnsmith/multimodal-food-ai)

---

## Overview

Cross-modal food understanding — connecting what food looks like to what it tastes like and how to make it — is a rich research problem with direct practical value in dietary tracking, recipe discovery, and food logging. This project builds a production-oriented implementation of cross-modal retrieval, extending and modernising the approach introduced by Salvador et al. (CVPR 2017) with contemporary tools.

Where the original im2recipe system used Torch7, Word2Vec, and skip-thought vectors, this implementation uses PyTorch, sentence-transformers, and CLIP — tools that are simpler to deploy, better documented, and more directly applicable to production systems. The text corpus is also larger: RecipeNLG at 2.2M recipes versus Recipe1M's 1M.

This project is framed not as a replication of published research but as a production-grade reimplementation using a modern stack — the kind of work applied ML engineers do when taking a research idea and making it deployable.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Image pathway                   │
│                                                  │
│  Food photo → CLIP ViT-B/32 → image embedding   │
│            → EfficientNet-B4 (fine-tuned)        │
│              → food category prediction          │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Joint embedding      │
         │   space (512-dim)      │
         │   FAISS index          │
         └────────────┬───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│                  Text pathway                    │
│                                                  │
│  Recipe text → sentence-transformers             │
│    (all-MiniLM-L6-v2)                            │
│  Title + ingredients + instructions → embedding  │
│  2.2M recipes indexed in FAISS                   │
└─────────────────────────────────────────────────┘
                      │
                      ▼
         Retrieval: image→recipe · recipe→image
         Evaluation: medR · R@1 · R@5 · R@10
```

---

## Key Features

- **CLIP-based image understanding** — zero-shot food image embeddings using OpenAI CLIP ViT-B/32, fine-tuned on Recipe1M food categories
- **EfficientNet food classifier** — EfficientNet-B4 fine-tuned on Recipe1M training images for food category and difficulty prediction
- **Large-scale text index** — sentence-transformer embeddings over 2.2M RecipeNLG recipes, indexed in FAISS for sub-second retrieval
- **Bidirectional retrieval** — image→recipe and recipe→image retrieval with medR, R@1, R@5, R@10 evaluation metrics
- **Self-supervised pairing** — semantic image-recipe pairing via embedding similarity, without requiring exact recipe-image alignment files
- **Production API** — FastAPI endpoint for image upload and recipe retrieval
- **Interactive demo** — Gradio interface for live image-to-recipe and recipe-to-image search

---

## Tech Stack

| Component | Technology |
|---|---|
| Image embeddings | OpenAI CLIP (`ViT-B/32`) |
| Image classifier | EfficientNet-B4 (PyTorch, fine-tuned) |
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | FAISS (flat L2) |
| Image corpus | Recipe1M (MIT CSAIL, ~800k food images) |
| Text corpus | RecipeNLG (2.2M recipes) |
| Training | Google Colab (A100 GPU) |
| Backend | FastAPI, Python 3.11 |
| Demo UI | Gradio |
| Deployment | Docker, Hugging Face Spaces |

---

## Project Structure

```
multimodal-food-ai/
├── data/
│   ├── sample_images/             # Sample extracted Recipe1M images
│   └── sample_recipes.csv         # Sample RecipeNLG text for reproducibility
├── embeddings/
│   ├── clip_encoder.py            # CLIP image embedding pipeline
│   ├── text_encoder.py            # Sentence-transformer recipe encoding
│   └── pairing.py                 # Semantic image-recipe pairing
├── models/
│   ├── efficientnet_classifier.py # EfficientNet fine-tuning
│   └── joint_embedding.py         # Joint embedding space training
├── retrieval/
│   ├── index.py                   # FAISS index builder
│   └── search.py                  # Bidirectional retrieval + evaluation
├── evaluation/
│   └── metrics.py                 # medR, R@1, R@5, R@10
├── api/
│   └── main.py                    # FastAPI image upload + retrieval
├── demo/
│   └── gradio_app.py              # Interactive Gradio interface
├── notebooks/
│   ├── 01_clip_embeddings.ipynb   # CLIP feature extraction
│   ├── 02_efficientnet_finetune.ipynb
│   └── 03_retrieval_evaluation.ipynb
├── deployment/
│   ├── Dockerfile
│   └── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/drnsmith/multimodal-food-ai.git
cd multimodal-food-ai
python -m venv venv && source venv/bin/activate
pip install -r deployment/requirements.txt

# Build CLIP embeddings for sample images
python embeddings/clip_encoder.py --images data/sample_images/ --out embeddings/image_index

# Build text embeddings for sample recipes
python embeddings/text_encoder.py --data data/sample_recipes.csv --out embeddings/text_index

# Run retrieval demo
python retrieval/search.py --query "pasta carbonara" --mode text2image

# Launch Gradio demo
python demo/gradio_app.py
```

### Colab Training

Full EfficientNet fine-tuning and joint embedding training notebooks are designed to run on Google Colab with A100 GPU. See `notebooks/` for step-by-step training pipelines.

---

## Evaluation

Evaluated on a held-out subset using standard cross-modal retrieval metrics:

| Model | medR ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|
| Random baseline | 500 | 0.001 | 0.005 | 0.010 |
| TF-IDF text matching | 42.3 | 0.08 | 0.19 | 0.27 |
| CLIP zero-shot | 18.6 | 0.15 | 0.34 | 0.46 |
| CLIP + sentence-transformers (ours) | TBD | TBD | TBD | TBD |

*Full evaluation in progress. Results will be updated as training completes.*

---

## Datasets

**Recipe1M+** — Salvador et al. (2017). *Learning Cross-modal Embeddings for Cooking Recipes and Food Images.* CVPR 2017. MIT CSAIL. Access via [im2recipe.csail.mit.edu](http://im2recipe.csail.mit.edu/)

**RecipeNLG** — Bień et al. (2020). *RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation.* INLG 2020. Available at [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/)

---

## Relation to Prior Work

This project is inspired by Salvador et al. (CVPR 2017) but is not a replication. Key differences:

| | Salvador et al. (2017) | This project |
|---|---|---|
| Framework | Torch7 (deprecated) | PyTorch 2.x |
| Image model | ResNet-50 | EfficientNet-B4 + CLIP |
| Text model | Skip-thoughts + Word2Vec | Sentence-transformers |
| Text corpus | Recipe1M (1M) | RecipeNLG (2.2M) |
| Pairing | Exact recipe-image alignment | Semantic similarity pairing |
| Deployment | Research only | FastAPI + HF Spaces |

---

## Related Projects

- [Recipe-Intelligence-Platform](https://github.com/drnsmith/recipe-intelligence-platform) — text-only pipeline (v1)
- [AI-Recipe-Classifier](https://github.com/drnsmith/recipe-difficulty-classifier) — difficulty classification benchmark

---

## Credits

Built by [@drnsmith](https://github.com/drnsmith) — quantitative data scientist specialising in multimodal ML and production AI systems.

[Medium](https://medium.com/@NeverOblivious) · [Substack](https://substack.com/@errolog) · [LinkedIn](https://linkedin.com/in/drnsmith)
