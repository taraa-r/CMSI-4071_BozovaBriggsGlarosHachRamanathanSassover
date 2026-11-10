# Image Model — Base Models and Datasets

Source and license record for the image classifier (SRS 5.3.4.3, 5.3.4.7).
Owners: Noa, Tara. Decided: November 9, 2026.

## Label convention

**0 = real (human-captured or human-made), 1 = AI-generated**, the same as the text model. OpenFake labels images with the strings `"real"` and `"fake"`: map `"real"` → 0 and `"fake"` → 1 on load.

## Base models

| Role | Model | License | Revision (Hugging Face commit) | How it's trained | Why |
|---|---|---|---|---|---|
| **Primary** | [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) | MIT ([openai/CLIP](https://github.com/openai/CLIP/blob/main/LICENSE)) | `57c216476eefef5ab752ec549e440a49ae4ae5f3` | Image encoder frozen; only a small classification layer is trained on its features | ViT design pretrained on ~400M image–caption pairs from the internet. CLIP-based detectors are reported to generalize better to generators not seen in training. Features are computed once, so training is cheap. |
| Comparison | [`google/efficientnet-b0`](https://huggingface.co/google/efficientnet-b0) | Apache-2.0 | `3d8d75d0812b65064f0151b456f869c6344bf20c` | Fully fine-tuned | Small, fast CNN pretrained on ImageNet. Works differently from CLIP (sliding filters, fine pixel traces), which makes it a useful contrast. Expected to be weaker on unseen generators and on compressed web images — the comparison tests that. |

Keep whichever does better on **validation** data, including the held-out generators (see below).

Considered and not chosen: [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) (Apache-2.0) — the same ViT design as CLIP's image encoder but trained only on labeled ImageNet-21k categories; CLIP vs. EfficientNet is the more informative comparison.

## Dataset

| Role | Dataset | License | Revision (Hugging Face commit) | Size | Notes |
|---|---|---|---|---|---|
| **Train / validation / test** | [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`ComplexDataLab/OpenFake`, subset `core`) | CC-BY-NC-4.0 (non-commercial use only) | `00b2f0ae3f5f7684c4b00f794f4c500076a9355d` | 2.3M train · 58.5k validation · 91.4k test | 75 generators, including 2025–2026 models (e.g. `flux.2-dev`, `imagen-4.0`). Real images from LAION and Pexels. Each AI image records its generator (`model`) and `release_date`. Use a sample — the full set is far too large. **Pin the revision:** the dataset was still being updated in November 2026. |

Considered and not chosen:
- [GenImage](https://github.com/GenImage-Dataset/GenImage) (CC BY-NC-SA 4.0) — older generators (Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM, BigGAN), real images only from ImageNet, distributed as very large archives via Google Drive / Baidu. Real and AI images are reported to differ in file format (a shortcut a model can learn). Possible extra test set only.
- [ELSA D3](https://huggingface.co/datasets/elsaEU/ELSA_D3) — no license listed on its page; real images are only URLs that must be downloaded from the web.

## Held-out generators

To measure how the model handles generators it never trained on, pick a few OpenFake generators (ideally including the newest ones) and **remove them from training**. Report results on them separately in validation and test. Record which generators were held out in the data manifest.

## Preprocessing (to match the extension and SRS 5.5.3)

- Only JPEG, PNG, and static WebP; width and height 32–4096 pixels; at most 16,000,000 decoded pixels; at most 10 MiB encoded.
- **Check for shortcuts first:** compare real vs. AI images by file format, JPEG quality, and size. If they differ, even it out — e.g. re-encode every training image as JPEG at random quality and randomly resize — so the model can't learn "PNG = AI" or "large = AI."
- Remove exact duplicates (OpenFake provides `image_sha256`), and make sure no image appears in more than one split.

## Known limitations

- CC-BY-NC-4.0 allows non-commercial use only; a commercial RealEye would need different training data.
- OpenAI's model card says CLIP was built for research and not for general deployment without careful study of the specific use.
- New generators appear constantly; results describe the held-out test data, not every image on the web (SRS 5.5.3).
