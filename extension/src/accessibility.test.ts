/// <reference types="node" />
import assert from "node:assert/strict";
import test from "node:test";

import {
  HELP_STEPS,
  POPUP_MODE_COPY,
  RESULT_MEANINGS,
  isAccessibleTargetSize,
  isKeyboardActivation,
} from "./accessibility.ts";

test("home scan options have the required accessible copy", () => {
  assert.equal(POPUP_MODE_COPY.screen.title, "Screen Scan");
  assert.equal(
    POPUP_MODE_COPY.screen.description,
    "Analyze text visible on this page.",
  );
  assert.equal(POPUP_MODE_COPY.selected.title, "Selected Content");
  assert.equal(
    POPUP_MODE_COPY.selected.description,
    "Analyze only the text you select.",
  );
});

test("Help includes the complete numbered workflow", () => {
  assert.equal(HELP_STEPS.length, 7);
  assert.ok(HELP_STEPS[0].startsWith("Open a webpage"));
  assert.ok(HELP_STEPS[5].includes("Open Explanation"));
  assert.ok(HELP_STEPS[6].includes("Dismiss warnings"));
});

test("Help explains every result category", () => {
  assert.ok(RESULT_MEANINGS.possibleAi.includes("warning"));
  assert.ok(RESULT_MEANINGS.noAi.includes("no configured warning"));
  assert.ok(RESULT_MEANINGS.inconclusive.includes("strong"));
  assert.ok(RESULT_MEANINGS.skipped.includes("200–5000"));
});

test("major action targets meet the 24-pixel minimum", () => {
  assert.equal(isAccessibleTargetSize(44, 44), true);
  assert.equal(isAccessibleTargetSize(24, 24), true);
  assert.equal(isAccessibleTargetSize(20, 44), false);
});

test("native action keys are recognized for keyboard users", () => {
  assert.equal(isKeyboardActivation("Enter"), true);
  assert.equal(isKeyboardActivation(" "), true);
  assert.equal(isKeyboardActivation("Tab"), false);
});
