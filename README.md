# CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover
Senior Project — **RealEye**, a Chrome extension that helps users spot potentially AI-generated text and images.

| Folder | What it is | SRS component |
|---|---|---|
| `extension/` | Chrome extension (Manifest V3, React 18 + TypeScript, built with Vite) | Extension Interface + Page Interaction CSCs |
| `server/` | Analysis API (Python + FastAPI) with the text model | Server CSC |
| `ml/` | Dataset prep, fine-tuning, evaluation (PyTorch + Hugging Face) | Evaluation CSC |
| `docs/` | `api-contract.md` (extension ↔ server format), `progress.md` (what's built so far) | |

> The server's `/analyze` scores text with the trained, calibrated text model (see **Get the text model** below). Image analysis, C2PA, and SynthID are not connected yet, so they report "unavailable".

---

## One-time setup

You need:
- **Git**
- **Node.js 22 or newer** — https://nodejs.org (check: `node -v`)
- **Python 3.11 or 3.12** from https://www.python.org/downloads/ (check: `python3.12 --version`). On a Mac, use the python.org installer: it runs natively on Apple Silicon. PyTorch 2.6 will not install on an Intel (Rosetta) Python, such as an old Intel Homebrew `python3.11`.
- **Google Chrome**

Then clone the repo (or `git pull` if you already have it):

```
git clone https://github.com/taraa-r/CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover.git
cd CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover
```

### 1. Server

**Windows (PowerShell):**
```
cd server
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Mac (Terminal):**
```
cd server
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
The install downloads PyTorch (about 1 GB), so it takes a few minutes. If you already have an older `server/.venv`, delete it first (`rm -rf .venv`).

### 1b. Get the text model

The trained model is about 330 MB, so it is **not in git**. Download it once from the team's Google Drive:

1. Open **My Drive → RealEye → models → distilroberta-base-20261030-0247** (add a shortcut to the shared `RealEye` folder in your My Drive if needed).
2. Create the folder `server/models/distilroberta-base-20261030-0247/` in the repo.
3. Download these files into it: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `calibration.json`. (The `.csv` and `.png` files are not needed.)

`server/models/` is in `.gitignore`. To use a model somewhere else, set `REALEYE_TEXT_MODEL_DIR` to its folder before starting the server. Without the model, the server still starts, but every text comes back "Unable to assess".

### 2. Extension

From the repo root (same on Windows and Mac):
```
cd extension
npm install
npm run build
```
This creates `extension/dist/` (not stored in git — everyone builds their own).

### 3. Load the extension into Chrome

1. Go to `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked** → select the `extension/dist` folder
4. Click the puzzle piece 🧩 in the toolbar → pin 📌 **RealEye**

---

## Running it (every time)

**Start the server** — from the `server` folder:

- Windows: `.venv\Scripts\python -m uvicorn app.main:app --reload`
- Mac: `.venv/bin/python -m uvicorn app.main:app --reload`

Keep that terminal open. Check it at http://127.0.0.1:8000/health → `"model_loaded": true` and `"model_version": "distilroberta-base-20261030-0247"`. If `model_loaded` is `false`, see **1b. Get the text model**.
Try the API in the browser at http://127.0.0.1:8000/docs.

**Use the extension:** open a page with paragraphs (e.g. a Wikipedia article) → click the RealEye icon → **Start scan**.
You should see each paragraph with **Possible AI content** or **No AI signs found**, its *Model estimate*, and a short explanation.

**After changing extension code:** run `npm run build` in `extension/`, then click reload ↻ on the RealEye card in `chrome://extensions`.

---

## Troubleshooting

**pip fails with `CERTIFICATE_VERIFY_FAILED`** — antivirus (Norton, McAfee, etc.) is scanning HTTPS traffic and re-signing it with its own certificate, which Python doesn't trust.
Either turn off the antivirus's encrypted-traffic scanning / exclude Python, or ask a teammate (Noa) for the project-only fix: export the antivirus root certificate, append it to a copy of pip's CA bundle, and point only `server/.venv/pip.ini` at it. Don't turn off SSL verification.

**"Service unavailable" in the popup** — the server isn't running. Start it (see above).

**"Chrome does not allow RealEye to scan this page"** — expected on `chrome://` pages and the Chrome Web Store.

**Developer mode is greyed out** — your Chrome is managed by an organization. Use a personal Chrome profile.

**"No text blocks of 200–5000 characters found"** — the page has no long paragraphs. The SRS minimum is 200 characters.

---

*Parts of this setup and code were written with help from Claude (Anthropic's AI assistant), per the course requirement to acknowledge AI assistance.*
