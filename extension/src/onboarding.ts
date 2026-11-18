export const ONBOARDING_KEY = "realeye_tutorial_completed";
export const UPLOAD_NOTICE_KEY = "realeye_upload_notice_acknowledged";

export type OnboardingSettings = {
  onboardingCompleted: boolean;
  uploadNoticeAcknowledged: boolean;
};

export const TUTORIAL_STEPS = [
  {
    title: "Welcome to RealEye",
    body: "RealEye helps identify text that may be AI-generated. Results are estimates, not definitive proof.",
  },
  {
    title: "Screen Scan",
    body: "Screen Scan analyzes eligible text currently visible in your browser viewport. Scroll to another part of the page and scan again to check more content.",
  },
  {
    title: "Selected Content",
    body: "Before opening RealEye, select or highlight specific text on the webpage. Selected Content analyzes only that selected text, not the rest of the page.",
  },
  {
    title: "Understanding Results",
    body: "Possible AI shows a warning result. No AI signs found means no configured warning triggered. Inconclusive means checks did not produce a strong result. Skipped text was outside the 200–5000 character range. Use Open Explanation for details and Dismiss to remove one warning.",
  },
] as const;

export function shouldShowTutorial(settings: OnboardingSettings): boolean {
  return !settings.onboardingCompleted;
}

export function shouldShowUploadNotice(settings: OnboardingSettings): boolean {
  return !settings.uploadNoticeAcknowledged;
}

export async function readOnboardingSettings(): Promise<OnboardingSettings> {
  if (typeof chrome === "undefined" || !chrome.storage?.sync) {
    return { onboardingCompleted: false, uploadNoticeAcknowledged: false };
  }
  const values = await chrome.storage.sync.get([ONBOARDING_KEY, UPLOAD_NOTICE_KEY]);
  return {
    onboardingCompleted: values[ONBOARDING_KEY] === true,
    uploadNoticeAcknowledged: values[UPLOAD_NOTICE_KEY] === true,
  };
}

export async function setTutorialCompleted(): Promise<void> {
  if (typeof chrome !== "undefined" && chrome.storage?.sync) {
    await chrome.storage.sync.set({ [ONBOARDING_KEY]: true });
  }
}

export async function setUploadNoticeAcknowledged(): Promise<void> {
  if (typeof chrome !== "undefined" && chrome.storage?.sync) {
    await chrome.storage.sync.set({ [UPLOAD_NOTICE_KEY]: true });
  }
}

export function scanMayBegin(settings: OnboardingSettings, consentAccepted: boolean): boolean {
  return consentAccepted || settings.uploadNoticeAcknowledged;
}

export function shouldStartPendingMode(pendingMode: "screen" | "selected", consentAccepted: boolean): "screen" | "selected" | null {
  return consentAccepted ? pendingMode : null;
}
