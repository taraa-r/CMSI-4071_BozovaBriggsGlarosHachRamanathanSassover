# CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover
Senior Project — **RealEye**, a Chrome extension that helps users spot potentially AI-generated text and images.

| Folder | What it is | SRS component |
|---|---|---|
| `extension/` | Chrome extension (Manifest V3, React 18 + TypeScript, built with Vite) | Extension Interface + Page Interaction CSCs |
| `server/` | Analysis API (Python 3.11 + FastAPI) | Server CSC |
| `ml/` | Dataset prep, fine-tuning, evaluation (PyTorch + Hugging Face) | Evaluation CSC |
| `docs/` | `api-contract.md` (extension ↔ server format), `progress.md` (what's built so far) | |

> The server's `/analyze` is currently **fake**: it returns the right format but always a 50% estimate. Real models replace it later.

---

## One-time setup

You need:
- **Git**
- **Node.js 22 or newer** — https://nodejs.org (check: `node -v`)
- **Python 3.11** — https://www.python.org/downloads/release/python-3119/ (PyTorch does not support newer Pythons yet)
- **Google Chrome**

Then clone the repo and switch to the branch:

```
git clone https://github.com/taraa-r/CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover.git
cd CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover
git checkout setup/project-skeleton
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
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

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

Keep that terminal open. Check it at http://127.0.0.1:8000/health → `{"status":"ok"}`.
Try the API in the browser at http://127.0.0.1:8000/docs.

**Use the extension:** open a page with paragraphs (e.g. a Wikipedia article) → click the RealEye icon → **Start scan**.
You should see "N text items analyzed" and entries like *Inconclusive — Model estimate: 50%*.

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
