import { useMemo, useRef, useState } from "react";
import { analyze, type ItemResult } from "./api";
import {
  applyRealEyeHighlights,
  clearRealEyeHighlights,
  extractTextBlocks,
  extractVisibleTextBlocks,
} from "./extract";

type ScanMode = "screen" | "selected";

type Status =
  | { kind: "idle" }
  | { kind: "scanning" }
  | { kind: "done"; results: ItemResult[]; texts: Map<string, string> }
  | { kind: "error"; message: string };

const STATE_LABELS: Record<ItemResult["overall_state"], string> = {
  warning: "Possible AI content",
  inconclusive: "No AI signs found",
  unable_to_analyze: "Unable to assess",
};

const EVIDENCE_LABELS: Record<string, string> = {
  classifier: "Model estimate",
  c2pa: "Content credentials",
  synthid: "Digital watermark",
};

const styles = `
  :root {
    color-scheme: light;
    font-family: Inter, "Segoe UI", sans-serif;
    background: #f5f7fb;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: #f5f7fb;
  }

  #root {
    width: 360px;
    min-height: 420px;
    background: linear-gradient(180deg, #f9fbff 0%, #eef3fa 100%);
  }

  .popup-shell {
    width: 360px;
    min-height: 420px;
    padding: 16px 16px 14px;
    color: #12263f;
    background: linear-gradient(180deg, #f9fbff 0%, #eef3fa 100%);
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .brand-mark {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: linear-gradient(135deg, #1d5ef0 0%, #5cd0ff 100%);
    box-shadow: 0 8px 18px rgba(29, 94, 240, 0.25);
  }

  .brand-mark svg {
    width: 18px;
    height: 18px;
    display: block;
  }

  .brand-copy {
    display: flex;
    flex-direction: column;
  }

  .eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b6f8a;
  }

  .brand-copy h1 {
    margin: 0;
    font-size: 20px;
    line-height: 1.2;
    letter-spacing: -0.03em;
    color: #13283f;
  }

  .pill {
    border: 1px solid rgba(29, 94, 240, 0.18);
    background: rgba(29, 94, 240, 0.06);
    color: #184dbd;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 6px 8px;
  }

  .hero-card,
  .summary-card,
  .state-card,
  .result-card,
  .error-card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(18, 38, 63, 0.08);
    border-radius: 16px;
    box-shadow: 0 10px 26px rgba(15, 33, 52, 0.05);
  }

  .hero-card {
    padding: 14px;
    margin-bottom: 12px;
  }

  .hero-card h2 {
    margin: 0 0 8px;
    font-size: 24px;
    line-height: 1.2;
    letter-spacing: -0.04em;
  }

  .hero-card p {
    margin: 0;
    font-size: 12px;
    line-height: 1.5;
    color: #53657f;
  }

  .mode-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 14px;
  }

  .mode-option {
    width: 100%;
    text-align: left;
    border: 1px solid rgba(18, 38, 63, 0.08);
    background: #ffffff;
    border-radius: 12px;
    padding: 12px 12px 10px;
    cursor: pointer;
    color: #17304c;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
  }

  .mode-option.selected {
    border-color: rgba(29, 94, 240, 0.35);
    box-shadow: 0 8px 18px rgba(29, 94, 240, 0.1);
    background: linear-gradient(180deg, rgba(29, 94, 240, 0.04), rgba(29, 94, 240, 0.01));
  }

  .mode-option:hover {
    border-color: rgba(29, 94, 240, 0.22);
    transform: translateY(-1px);
  }

  .mode-title {
    display: block;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
  }

  .mode-description {
    display: block;
    font-size: 11px;
    line-height: 1.45;
    color: #5d6f84;
  }

  .info-banner {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 10px;
    background: #edf4ff;
    border: 1px solid rgba(29, 94, 240, 0.1);
    color: #214e9f;
    font-size: 11px;
    line-height: 1.5;
    font-weight: 600;
  }

  .action-button,
  .secondary-button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.01em;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
  }

  .action-button {
    background: linear-gradient(135deg, #0f2f6a 0%, #1d5ef0 100%);
    color: white;
    box-shadow: 0 10px 20px rgba(29, 94, 240, 0.25);
  }

  .secondary-button {
    background: #edf3fb;
    color: #193d63;
    border: 1px solid rgba(18, 38, 63, 0.08);
  }

  .action-button:hover:not(:disabled),
  .secondary-button:hover:not(:disabled) {
    transform: translateY(-1px);
  }

  .action-button:hover:not(:disabled) {
    transform: translateY(-1px);
  }

  .action-button:disabled {
    opacity: 0.75;
    cursor: wait;
  }

  .scanning-card,
  .error-card {
    margin-top: 12px;
    padding: 14px;
  }

  .status-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2px solid rgba(29, 94, 240, 0.24);
    border-top-color: #1d5ef0;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .status-copy {
    font-size: 13px;
    font-weight: 600;
    color: #1a3553;
  }

  .microcopy {
    margin: 0;
    font-size: 12px;
    line-height: 1.5;
    color: #5d6f84;
  }

  .error-card {
    border-color: rgba(181, 42, 42, 0.12);
    background: rgba(255, 249, 249, 0.9);
    margin-top: 12px;
  }

  .error-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 8px;
    font-size: 13px;
    font-weight: 700;
    color: #8b2a2a;
  }

  .error-card p {
    margin: 0;
    font-size: 12px;
    line-height: 1.5;
    color: #6d2d2d;
  }

  .results-wrap {
    margin-top: 12px;
  }

  .summary-card {
    padding: 14px;
    margin-bottom: 10px;
  }

  .summary-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }

  .summary-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #2dbd7a;
    box-shadow: 0 0 0 6px rgba(45, 189, 122, 0.12);
  }

  .summary-dot.warning {
    background: #f3a33a;
    box-shadow: 0 0 0 6px rgba(243, 163, 58, 0.12);
  }

  .summary-title h3 {
    margin: 0;
    font-size: 15px;
    letter-spacing: -0.03em;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
  }

  .summary-stat {
    padding: 8px 10px;
    border-radius: 10px;
    background: #f2f6fb;
    border: 1px solid rgba(18, 38, 63, 0.06);
  }

  .summary-stat strong {
    display: block;
    font-size: 15px;
    line-height: 1.1;
    color: #162d4b;
  }

  .summary-stat span {
    display: block;
    margin-top: 4px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #557089;
  }

  .result-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .result-card {
    padding: 10px 10px 8px;
  }

  .result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 5px 8px;
    border: 1px solid transparent;
  }

  .badge.warning {
    background: rgba(243, 163, 58, 0.12);
    border-color: rgba(243, 163, 58, 0.18);
    color: #99621d;
  }

  .badge.inconclusive {
    background: rgba(31, 120, 255, 0.08);
    border-color: rgba(31, 120, 255, 0.14);
    color: #214e9f;
  }

  .badge.unable_to_analyze {
    background: rgba(86, 99, 112, 0.08);
    border-color: rgba(86, 99, 112, 0.14);
    color: #44556a;
  }

  .confidence {
    font-size: 10px;
    font-weight: 700;
    line-height: 1.2;
    color: #405671;
    text-align: right;
  }

  .meter {
    margin-top: 8px;
    width: 100%;
    height: 7px;
    border-radius: 999px;
    background: #eaf0f5;
    overflow: hidden;
  }

  .meter > span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #1d5ef0 0%, #7dd4ff 100%);
  }

  .meter.warning > span {
    background: linear-gradient(90deg, #f39d35 0%, #f9c770 100%);
  }

  .snippet {
    margin: 0;
    font-size: 11px;
    line-height: 1.5;
    color: #4d6280;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    overflow: hidden;
  }

  .caveat {
    margin: 6px 0 0;
    text-align: right;
    font-size: 10px;
    font-style: italic;
    color: #5d6f84;
  }

  .meta {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tag {
    display: inline-flex;
    align-items: center;
    padding: 5px 7px;
    border-radius: 999px;
    font-size: 9px;
    line-height: 1.1;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: #edf4ff;
    color: #2f5a9c;
  }

  .tag.warning {
    background: rgba(243, 163, 58, 0.12);
    color: #9a6326;
  }

  .tag.neutral {
    background: #edf4ff;
    color: #2f5a9c;
  }

  .disclaimer {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid rgba(18, 38, 63, 0.08);
    font-size: 10px;
    line-height: 1.5;
    color: #58708a;
  }
`;

export default function Popup() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [scanMode, setScanMode] = useState<ScanMode>("screen");
  const activeScanRef = useRef(0);

  const summary = useMemo(() => {
    if (status.kind !== "done") return null;
    const counts = {
      warning: status.results.filter((result) => result.overall_state === "warning").length,
      inconclusive: status.results.filter(
        (result) => result.overall_state === "inconclusive",
      ).length,
      unable: status.results.filter(
        (result) => result.overall_state === "unable_to_analyze",
      ).length,
    };

    return {
      ...counts,
      hasWarnings: counts.warning > 0,
    };
  }, [status]);

  async function clearPageHighlights() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) return;
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: clearRealEyeHighlights,
      });
    } catch {
      // Ignore page highlight cleanup when the page context is unavailable.
    }
  }

  function cancelScan() {
    activeScanRef.current += 1;
    void clearPageHighlights();
    setStatus({ kind: "idle" });
  }

  function closeResults() {
    activeScanRef.current += 1;
    void clearPageHighlights();
    if (typeof window !== "undefined" && typeof window.close === "function") {
      window.close();
      return;
    }
    setStatus({ kind: "idle" });
  }

  async function startScan() {
    const scanToken = ++activeScanRef.current;
    setStatus({ kind: "scanning" });
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id || !tab.url) throw new Error("No active tab found.");

      const extractionFunction = (scanMode === "screen"
        ? extractVisibleTextBlocks
        : extractTextBlocks) as (...args: unknown[]) => unknown;
      let extractedBlocks: Array<{ id: string; text: string; selector: string }> | string[];
      try {
        const [injection] = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: extractionFunction,
        });
        extractedBlocks = (injection.result ?? []) as
          | Array<{ id: string; text: string; selector: string }>
          | string[];
      } catch {
        throw new Error("Chrome does not allow RealEye to scan this page.");
      }

      if (scanToken !== activeScanRef.current) return;

      const blocks = (Array.isArray(extractedBlocks)
        ? extractedBlocks
        : []
      ).map((block) =>
        typeof block === "string"
          ? { id: crypto.randomUUID(), text: block, selector: "" }
          : { id: block.id ?? crypto.randomUUID(), text: block.text ?? "", selector: block.selector ?? "" },
      );

      if (blocks.length === 0) {
        const noVisibleTextMessage = scanMode === "screen"
          ? "No eligible visible text was found in the current viewport. Scroll to the article content and try again."
          : "No text blocks of 200–5000 characters were found on this page.";
        throw new Error(noVisibleTextMessage);
      }

      const scanId = crypto.randomUUID();
      const textMap = new Map<string, string>();
      const markerMap = new Map<string, string>();
      const allResults: ItemResult[] = [];

      for (let index = 0; index < blocks.length; index += 10) {
        const batch = blocks.slice(index, index + 10);
        const items = batch.map((block) => {
          const itemId = crypto.randomUUID();
          textMap.set(itemId, block.text);
          markerMap.set(itemId, block.id);
          return { item_id: itemId, type: "text" as const, text: block.text };
        });

        const response = await analyze({
          scan_id: scanId,
          page_url: tab.url,
          scan_time: new Date().toISOString(),
          items,
        });

        allResults.push(...response.results);
      }

      if (scanToken !== activeScanRef.current) return;

      const highlightPayload = allResults.map((result) => ({
        item_id: result.item_id,
        domMarkerId: markerMap.get(result.item_id) ?? result.item_id,
        overall_state: result.overall_state,
        explanation: result.explanation,
      }));

      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: applyRealEyeHighlights,
        args: [highlightPayload],
      });

      setStatus({ kind: "done", results: allResults, texts: textMap });
    } catch (err) {
      if (scanToken !== activeScanRef.current) return;
      const message =
        err instanceof TypeError ? "Service unavailable" : (err as Error).message;
      setStatus({ kind: "error", message });
    }
  }

  return (
    <>
      <style>{styles}</style>
      <main className="popup-shell">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2.75C8.98 2.75 6.5 4.71 6.5 7.67V9.9c0 3.62 2.07 6.83 5.2 8.64a1.3 1.3 0 0 0 1.2 0c3.13-1.81 5.2-5.02 5.2-8.64V7.67c0-2.96-2.48-4.92-5.5-4.92Zm-1.1 8.2 1.41 1.41 3.09-3.1 1.1 1.1-4.2 4.2-2.5-2.5 1.1-1.1Z" fill="white"/>
              </svg>
            </div>
            <div className="brand-copy">
              <span className="eyebrow">AI content scan</span>
              <h1>RealEye</h1>
            </div>
          </div>
          <span className="pill">Browser</span>
        </header>

        {status.kind === "idle" && (
          <>
            <section className="hero-card">
              <h2>Choose a scan type</h2>
              <p>RealEye checks page text and highlights possible AI-generated patterns.</p>

              <div className="mode-list">
                <button
                  type="button"
                  className={`mode-option ${scanMode === "screen" ? "selected" : ""}`}
                  onClick={() => setScanMode("screen")}
                  aria-pressed={scanMode === "screen"}
                >
                  <span className="mode-title">Screen Scan</span>
                  <span className="mode-description">Scan the full page for likely AI-generated text.</span>
                </button>

                <button
                  type="button"
                  className={`mode-option ${scanMode === "selected" ? "selected" : ""}`}
                  onClick={() => setScanMode("selected")}
                  aria-pressed={scanMode === "selected"}
                >
                  <span className="mode-title">Selected Content</span>
                  <span className="mode-description">Check highlighted or chosen text on the page.</span>
                </button>
              </div>

              <div className="info-banner">
                First scan: RealEye sends page text to the local analysis service to check for AI-generated patterns.
              </div>
            </section>

            <button className="action-button" onClick={startScan}>
              Start scan
            </button>
          </>
        )}

        {status.kind === "scanning" && (
          <section className="scanning-card" aria-live="polite">
            <div className="status-row">
              <div className="spinner" aria-hidden="true" />
              <div className="status-copy">Scanning this page…</div>
            </div>
            <p className="microcopy">
              We are extracting readable text and checking the model signals behind each item.
            </p>
            <div style={{ marginTop: 12 }}>
              <button type="button" className="secondary-button" onClick={cancelScan}>
                Cancel
              </button>
            </div>
          </section>
        )}

        {status.kind === "error" && (
          <section className="error-card" role="alert">
            <div className="error-title">⚠ Could not finish this scan</div>
            <p>{status.message}</p>
            <div style={{ marginTop: 12 }}>
              <button className="action-button" onClick={startScan}>
                Try again
              </button>
            </div>
          </section>
        )}

        {status.kind === "done" && (
          <div className="results-wrap" aria-live="polite">
            <section className="summary-card">
              <div className="summary-title">
                <span
                  className={`summary-dot ${summary?.hasWarnings ? "warning" : ""}`}
                  aria-hidden="true"
                />
                <h3>
                  {summary?.hasWarnings
                    ? "Possible AI content detected"
                    : "No strong signal detected"}
                </h3>
              </div>

              <div className="summary-grid">
                <div className="summary-stat">
                  <strong>{status.results.length}</strong>
                  <span>Items</span>
                </div>
                <div className="summary-stat">
                  <strong>{summary?.warning ?? 0}</strong>
                  <span>Warning</span>
                </div>
                <div className="summary-stat">
                  <strong>{summary?.inconclusive ?? 0}</strong>
                  <span>No AI signs</span>
                </div>
              </div>
            </section>

            <ul className="result-list">
              {status.results.map((result) => {
                const preview = status.texts.get(result.item_id)?.trim().replace(/\s+/g, " ") ?? "";
                const estimate = result.classifier.estimate != null ? Math.round(result.classifier.estimate * 100) : null;
                const signalSummary = result.warning_evidence.length > 0
                  ? result.warning_evidence.map((signal) => EVIDENCE_LABELS[signal] ?? signal).join(" · ")
                  : "No signal triggered";

                return (
                  <li key={result.item_id} className="result-card">
                    <div className="result-top">
                      <span className={`badge ${result.overall_state}`}>
                        {STATE_LABELS[result.overall_state]}
                      </span>
                      {estimate != null && (
                        <span className="confidence">Model estimate: {estimate}%</span>
                      )}
                    </div>

                    {preview && <p className="snippet">“{preview.slice(0, 120)}”</p>}

                    {estimate != null && (
                      <div className={`meter ${result.overall_state === "warning" ? "warning" : ""}`}>
                        <span style={{ width: `${Math.min(100, Math.max(8, estimate))}%` }} />
                      </div>
                    )}

                    <div className="meta">
                      <span className={`tag ${result.overall_state === "warning" ? "warning" : "neutral"}`}>
                        {signalSummary}
                      </span>
                    </div>

                    {result.overall_state !== "unable_to_analyze" && (
                      <p className="caveat">Estimate only. May be inaccurate.</p>
                    )}
                  </li>
                );
              })}
            </ul>

            <div className="disclaimer">
              AI detection can be wrong. Missing credentials or watermarks do not prove that a
              person created this content.
            </div>

            <div style={{ marginTop: 12 }}>
              <button type="button" className="secondary-button" onClick={closeResults}>
                Close
              </button>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
