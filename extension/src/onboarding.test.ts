/// <reference types="node" />
import assert from "node:assert/strict";
import test from "node:test";

import {
  ONBOARDING_KEY,
  UPLOAD_NOTICE_KEY,
  TUTORIAL_STEPS,
  shouldShowTutorial,
  shouldShowUploadNotice,
  scanMayBegin,
  shouldStartPendingMode,
  readOnboardingSettings,
  setTutorialCompleted,
  setUploadNoticeAcknowledged,
} from "./onboarding.ts";

test("first-time users see a four-step tutorial", () => {
  assert.equal(TUTORIAL_STEPS.length, 4);
  assert.equal(shouldShowTutorial({ onboardingCompleted: false, uploadNoticeAcknowledged: false }), true);
});

test("returning users do not see the tutorial again", () => {
  assert.equal(shouldShowTutorial({ onboardingCompleted: true, uploadNoticeAcknowledged: false }), false);
});

test("tutorial and upload acknowledgement are separate settings", () => {
  assert.equal(shouldShowUploadNotice({ onboardingCompleted: true, uploadNoticeAcknowledged: false }), true);
  assert.equal(shouldShowTutorial({ onboardingCompleted: true, uploadNoticeAcknowledged: false }), false);
  assert.equal(shouldShowUploadNotice({ onboardingCompleted: false, uploadNoticeAcknowledged: true }), false);
});

test("Chrome sync storage persists onboarding and upload settings", async () => {
  const values: Record<string, unknown> = {};
  Object.assign(globalThis, {
    chrome: {
      storage: {
        sync: {
          get: async (keys: string[]) => Object.fromEntries(keys.map((key) => [key, values[key]])),
          set: async (items: Record<string, unknown>) => Object.assign(values, items),
        },
      },
    },
  });

  await setTutorialCompleted();
  await setUploadNoticeAcknowledged();
  assert.equal(values[ONBOARDING_KEY], true);
  assert.equal(values[UPLOAD_NOTICE_KEY], true);

  const settings = await readOnboardingSettings();
  assert.deepEqual(settings, { onboardingCompleted: true, uploadNoticeAcknowledged: true });
});

test("consent cancellation sends no scan and acceptance resumes the selected mode", () => {
  const settings = { onboardingCompleted: true, uploadNoticeAcknowledged: false };
  assert.equal(scanMayBegin(settings, false), false);
  assert.equal(shouldStartPendingMode("selected", false), null);
  assert.equal(scanMayBegin(settings, true), true);
  assert.equal(shouldStartPendingMode("selected", true), "selected");
  assert.equal(shouldStartPendingMode("screen", true), "screen");
});
