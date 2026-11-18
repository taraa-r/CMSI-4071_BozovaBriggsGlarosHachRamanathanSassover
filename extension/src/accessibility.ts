export const POPUP_MODE_COPY = {
  screen: {
    title: "Screen Scan",
    description: "Analyze text visible on this page.",
  },
  selected: {
    title: "Selected Content",
    description: "Analyze only the text you select.",
  },
} as const;

export const HELP_STEPS = [
  "Open a webpage you want to check.",
  "Click the RealEye extension icon.",
  "Choose Screen Scan or Selected Content.",
  "Wait while RealEye analyzes the content.",
  "Review the results and Possible AI warnings.",
  "Select Open Explanation for more details.",
  "Dismiss warnings or close the results when finished.",
] as const;

export const RESULT_MEANINGS = {
  possibleAi: "Possible AI indicates a warning result that matched a configured AI signal.",
  noAi: "No AI signs found means no configured warning signal was triggered.",
  inconclusive: "Inconclusive means the available checks did not produce a strong enough result.",
  skipped: "Skipped means the text was outside the required 200–5000 character range and was not sent to the backend.",
} as const;

export function isAccessibleTargetSize(width: number, height: number): boolean {
  return width >= 24 && height >= 24;
}

export function isKeyboardActivation(key: string): boolean {
  return key === "Enter" || key === " " || key === "Spacebar";
}
