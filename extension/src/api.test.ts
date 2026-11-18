/// <reference types="node" />
import assert from "node:assert/strict";
import test from "node:test";

import { analyze, friendlyApiError, ApiError } from "./api.ts";

const request = {
  scan_id: "b3f1c2e0-6a4d-4f7e-9c1a-2d8e5f0a7b11",
  page_url: "https://example.com/article",
  scan_time: "2026-11-17T12:00:00Z",
  items: [{ item_id: "4c7d2a91-0e3b-4f6a-8b5d-91e2c7f0a3d4", type: "text" as const, text: "A".repeat(220) }],
};

test("analyze accepts a successful backend response", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    scan_id: request.scan_id,
    rule_version: "rule-1.0",
    results: [{ item_id: request.items[0].item_id, overall_state: "inconclusive" }],
  }), { status: 200, headers: { "Content-Type": "application/json" } });

  try {
    const response = await analyze(request);
    assert.equal(response.results.length, 1);
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test("analyze treats a contract HTTP error as ApiError without exposing diagnostic text", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    error: { code: "invalid_request", message: "Lazy-loaded images should have explicit dimensions\n23 elements" },
  }), { status: 422, headers: { "Content-Type": "application/json" } });

  try {
    await assert.rejects(
      () => analyze(request),
      (error: unknown) => error instanceof ApiError && error.status === 422,
    );
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test("friendlyApiError hides raw webpage diagnostics for HTTP failures", () => {
  const message = friendlyApiError(new ApiError(422, "invalid_request", "Lazy-loaded images should have explicit dimensions"));
  assert.equal(message, "The server could not accept this analysis request.");
});

test("friendlyApiError gives a short offline-server message for network failures", () => {
  const message = friendlyApiError(new ApiError(null, null, "Service unavailable"));
  assert.equal(message, "Service unavailable. Start the RealEye backend and try again.");
});
