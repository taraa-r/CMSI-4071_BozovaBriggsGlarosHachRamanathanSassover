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

Keep whichever does better on **validation** data. Unseen generators are measured only at the end, on `core/test` (see below).

Considered and not chosen: [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) (Apache-2.0) — the same ViT design as CLIP's image encoder but trained only on labeled ImageNet-21k categories; CLIP vs. EfficientNet is the more informative comparison.

## Dataset

| Role | Dataset | License | Revision (Hugging Face commit) | Size | Notes |
|---|---|---|---|---|---|
| **Train / validation / test** | [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`ComplexDataLab/OpenFake`, subset `core`) | CC-BY-NC-4.0 (non-commercial use only) | `00b2f0ae3f5f7684c4b00f794f4c500076a9355d` | 2.3M train (3.3 TB) · 58.5k validation (72 GB) · 91.4k test (68 GB) | Dozens of generators, including 2025–2026 models (e.g. `flux.2-dev`, `imagen-4.0`); validation alone has 58. Real images: LAION + Pexels (train/validation), DOCCI + ImageNet (test). Each AI image records its generator (`model`) and `release_date`. Too large to download: read labels and sampled chunks directly from Hugging Face (`explore_openfake.py`). **Pin the revision:** the dataset was still being updated in November 2026. |
| Extra in-the-wild test (optional) | OpenFake subset `reddit` (`test` split) | CC-BY-NC-4.0 | same | 25 GB | AI images from AI-generation subreddits, real photos from photography subreddits, with real platform compression. Use only for a final check, never for training or choosing the cutoff. |

Considered and not chosen:
- [GenImage](https://github.com/GenImage-Dataset/GenImage) (CC BY-NC-SA 4.0) — older generators (Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM, BigGAN), real images only from ImageNet, distributed as very large archives via Google Drive / Baidu. Real and AI images are reported to differ in file format (a shortcut a model can learn). Possible extra test set only.
- [ELSA D3](https://huggingface.co/datasets/elsaEU/ELSA_D3) — no license listed on its page; real images are only URLs that must be downloaded from the web.

## Unseen generators: already built in

OpenFake's `core/test` split is out-of-distribution by design (dataset card): its AI images come only from generators **never used in train or validation** (e.g. `gpt-image-1.5`, `gpt-image-2.0`, `nano-banana-pro`, `flux.2-klein-9b`, `z-image-turbo`, `recraft-v2`/`v3`, `midjourney-7`, `ideogram-2.0`), and its real photos come from different sources (DOCCI + ImageNet). So the test score already measures unseen generators and unseen real photos; we don't need to hold generators out ourselves. Validation uses the same generators and real sources as training.

Do **not** add OpenFake's `supplement_*` subsets to training without filtering: some of their generators are `core/test` generators (dataset card).

## Preprocessing (to match the extension and SRS 5.5.3)

**Shortcut check (Nov 9, `explore_openfake.py`, 627 sampled validation images): real and AI images differ a lot, so a model could learn file properties instead of signs of AI.**

| | Real | AI |
|---|---|---|
| File format | 98% JPEG | 51% JPEG, 48% PNG |
| JPEG quality (median) | 83 | 95 |
| File size (median) | 105 KB | 612 KB |
| Image size (median) | 0.6 MP (LAION 0.33 MP, Pexels 23 MP) | 1.0 MP (often 1024×1024) |
| Has camera data (EXIF) | 8% | 15% |
| Within SRS image limits | 69% (only 14% of Pexels) | 99% |

So data prep must treat every image, real and AI, the same way:

- **Resize** to a similar size range, and shrink oversized images (e.g. Pexels) to within the SRS limits instead of dropping them: width and height 32–4096 pixels, at most 16,000,000 pixels.
- **Re-encode as JPEG at a random quality** (e.g. 70–95), which also removes PNG, file-size and EXIF differences. The extension sends JPEG, PNG and static WebP up to 10 MiB.
- Remove exact duplicates (57 in validation; OpenFake provides `image_sha256`). OpenFake states there are no overlaps between splits.
- Normalize the `type` column spellings (`base`/`Base`, `finetune`/`Fine-tune`, `lora`/`LoRA`).
- Look at `tiny-random-sana` images before training: the name suggests a tiny randomly initialized test model, so they may be noise rather than realistic AI images.

## Known limitations

- CC-BY-NC-4.0 allows non-commercial use only; a commercial RealEye would need different training data.
- OpenAI's model card says CLIP was built for research and not for general deployment without careful study of the specific use.
- New generators appear constantly; results describe OpenFake's test data, not every image on the web (SRS 5.5.3).
