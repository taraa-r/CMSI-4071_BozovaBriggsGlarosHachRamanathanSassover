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
    reason?: "length" | "language" | "not_available" | null;
    analyzed_chars?: number | null;
    total_chars?: number | null;
  };
  c2pa: { status: string };
  synthid: { status: string };
  limitations: string[];
  explanation: string;
}

export interface AnalyzeResponse {
  scan_id: string;
  rule_version: string;
  results: ItemResult[];
}

export class ApiError extends Error {
  readonly status: number | null;
  readonly code: string | null;

  constructor(status: number | null, code: string | null, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function friendlyApiError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status == null) return "Service unavailable. Start the RealEye backend and try again.";
    if (error.status === 429) return "Too many requests. Wait a moment and try again.";
    if (error.status === 413) return "The request was too large to analyze.";
    if (error.status === 422) return "The server could not accept this analysis request.";
    if (error.status >= 500) return "The RealEye server encountered an error. Please try again.";
    if (error.code === "no_items") return "No eligible text was available to analyze.";
    return "The analysis request failed. Please try again.";
  }

  if (error instanceof TypeError) {
    return "Service unavailable. Start the RealEye backend and try again.";
  }

  return "The analysis could not be completed. Please try again.";
}

export async function analyze(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new ApiError(null, null, "Service unavailable");
  }

  const bodyText = await response.text().catch(() => "");

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = bodyText ? JSON.parse(bodyText) : null;
    } catch {
      body = null;
    }

    const record = body as { error?: { code?: string; message?: string }; detail?: unknown } | null;
    const code = record?.error?.code ?? null;
    const message = record?.error?.message;

    throw new ApiError(response.status, code, typeof message === "string" ? message : `Server returned ${response.status}`);
  }

  let body: AnalyzeResponse;
  try {
    body = bodyText ? JSON.parse(bodyText) as AnalyzeResponse : null as unknown as AnalyzeResponse;
  } catch {
    throw new ApiError(response.status, "invalid_response", "The server returned an invalid response.");
  }

  if (!body || !Array.isArray(body.results)) {
    throw new ApiError(response.status, "invalid_response", "The server returned an invalid response.");
  }

  return body;
}
