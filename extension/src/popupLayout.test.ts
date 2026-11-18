/// <reference types="node" />
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import { POPUP_LAYOUT, isPopupWidthSupported } from "./popupLayout.ts";

test("popup uses a stable readable width within Chrome limits", () => {
  assert.equal(POPUP_LAYOUT.width, 400);
  assert.equal(isPopupWidthSupported(POPUP_LAYOUT.width), true);
});

test("popup height allows scrolling while keeping a usable minimum", () => {
  assert.equal(POPUP_LAYOUT.minHeight, 480);
  assert.equal(POPUP_LAYOUT.maxHeight, 520);
  assert.ok(POPUP_LAYOUT.maxHeight >= POPUP_LAYOUT.minHeight);
});

test("narrow zoom fallback is below the normal popup width", () => {
  assert.ok(POPUP_LAYOUT.narrowViewport < POPUP_LAYOUT.width);
});

test("index HTML establishes popup width before React mounts", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.ok(html.includes("width=400"));
  assert.ok(html.includes("width: 400px"));
  assert.ok(html.includes("max-height: 520px"));
  assert.ok(html.includes("overflow-x: hidden"));
});

test("compact home layout removes oversized descriptions and keeps cards within 520px", () => {
  const popup = readFileSync(new URL("./Popup.tsx", import.meta.url), "utf8");
  assert.ok(popup.includes("What would you like to scan?"));
  assert.ok(popup.includes("min-height: 480px"));
  assert.ok(popup.includes("min-height: 84px"));
  assert.ok(popup.includes(".hero-card p,\n  .hero-card .info-banner"));
});
