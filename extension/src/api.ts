// Types and client for the RealEye server. Must match docs/api-contract.md.

export const API_URL = "http://127.0.0.1:8000";

export type ItemType = "text" | "image";
export type OverallState = "warning" | "inconclusive" | "unable_to_analyze";

export interface AnalyzeItem {
  item_id: string;
  type: ItemType;
  text?: string;
  image_base64?: string;
  mime_type?: "image/jpeg" | "image/png" | "image/webp";
  source_url?: string;
}

export interface AnalyzeRequest {
  scan_id: string;
  page_url: string;
  scan_time: string;
  items: AnalyzeItem[];
}

export interface ItemResult {
  item_id: string;
  type: ItemType;
  overall_state: OverallState;
  conflict: boolean;
  warning_evidence: ("classifier" | "c2pa" | "synthid")[];
  classifier: {
    status: "completed" | "unsupported" | "error";
    estimate?: number | null;
    estimate_kind: "model_estimate";
    label?: "ai" | "human" | null;
    threshold?: number | null;
    model_version?: string | null;
  };
  c2pa: { status: string };
  synthid: { status: string };
  limitations: string[];
}

export interface AnalyzeResponse {
  scan_id: string;
  rule_version: string;
  results: ItemResult[];
}

export async function analyze(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error?.message ?? `Server returned ${response.status}`);
  }
  return response.json();
}
