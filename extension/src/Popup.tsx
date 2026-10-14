import { useState } from "react";
import { analyze, type ItemResult } from "./api";
import { extractTextBlocks } from "./extract";

type Status =
  | { kind: "idle" }
  | { kind: "scanning" }
  | { kind: "done"; results: ItemResult[]; texts: Map<string, string> }
  | { kind: "error"; message: string };

const STATE_LABELS: Record<ItemResult["overall_state"], string> = {
  warning: "Warning",
  inconclusive: "Inconclusive",
  unable_to_analyze: "Unable to analyze",
};

export default function Popup() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  async function startScan() {
    setStatus({ kind: "scanning" });
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id || !tab.url) throw new Error("No active tab found.");

      let blocks: string[];
      try {
        const [injection] = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: extractTextBlocks,
        });
        blocks = injection.result ?? [];
      } catch {
        throw new Error("Chrome does not allow RealEye to scan this page.");
      }

      if (blocks.length === 0) {
        throw new Error("No text blocks of 200–5000 characters found on this page.");
      }

      const items = blocks.map((text) => ({
        item_id: crypto.randomUUID(),
        type: "text" as const,
        text,
      }));
      const response = await analyze({
        scan_id: crypto.randomUUID(),
        page_url: tab.url,
        scan_time: new Date().toISOString(),
        items,
      });

      const texts = new Map(items.map((item) => [item.item_id, item.text]));
      setStatus({ kind: "done", results: response.results, texts });
    } catch (err) {
      // fetch throws TypeError when the server can't be reached (SRS 5.3.1.44).
      const message =
        err instanceof TypeError ? "Service unavailable" : (err as Error).message;
      setStatus({ kind: "error", message });
    }
  }

  return (
    <main style={{ width: 360, padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 20, margin: "0 0 12px" }}>RealEye</h1>

      <button onClick={startScan} disabled={status.kind === "scanning"}>
        Start scan
      </button>

      {status.kind === "scanning" && <p role="status">Scanning...</p>}

      {status.kind === "error" && <p role="alert">{status.message}</p>}

      {status.kind === "done" && (
        <>
          <p role="status">{status.results.length} text items analyzed.</p>
          <ul style={{ paddingLeft: 16 }}>
            {status.results.map((r) => (
              <li key={r.item_id} style={{ marginBottom: 10 }}>
                <strong>{STATE_LABELS[r.overall_state]}</strong>
                {r.classifier.estimate != null && (
                  <> — Model estimate: {Math.round(r.classifier.estimate * 100)}%</>
                )}
                <div style={{ fontSize: 12, color: "#555" }}>
                  “{status.texts.get(r.item_id)?.slice(0, 80)}…”
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </main>
  );
}
