import { HELP_STEPS, POPUP_MODE_COPY, RESULT_MEANINGS } from "./accessibility";
import { useEffect, useMemo, useRef, useState } from "react";
import { analyze, friendlyApiError, type ItemResult } from "./api";
import {
  readOnboardingSettings,
  setTutorialCompleted,
  setUploadNoticeAcknowledged,
  TUTORIAL_STEPS,
  shouldShowTutorial,
  shouldShowUploadNotice,
  type OnboardingSettings,
} from "./onboarding";
import {
  applyRealEyeHighlights,
  clearRealEyeHighlights,
  extractVisibleTextBlocks,
  extractSelectedTextBlocks,
  sortRealEyeItems,
  paginateRealEyeItems,
  getPageHighlightPayload,
  isActiveScanToken,
  type SkippedDisplayItem,
  type ExtractedTextBlock,
} from "./extract";

type ScanMode = "screen" | "selected";

type Status =
  | { kind: "idle" }
  | { kind: "scanning" }
  | { kind: "done"; results: ItemResult[]; skipped: SkippedDisplayItem[]; texts: Map<string, string>; domMarkers: Map<string, string> }
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

  html,
  body,
  #root {
    width: 400px;
    max-width: 400px;
  }

  html,
  body {
    margin: 0;
    min-width: 400px;
    min-height: 420px;
    max-height: 600px;
    overflow-x: hidden;
  }

  body {
    background: #f5f7fb;
  }

  #root {
    min-height: 420px;
    max-height: 600px;
    overflow-x: hidden;
    overflow-y: auto;
    background: linear-gradient(180deg, #f9fbff 0%, #eef3fa 100%);
  }

  .popup-shell {
    width: 400px;
    min-height: 480px;
    max-height: 520px;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 10px 12px 12px;
    color: #12263f;
    background: linear-gradient(180deg, #f9fbff 0%, #eef3fa 100%);
  }

  .topbar {
    min-height: 52px;
    margin-bottom: 6px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .brand-mark {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #1d5ef0 0%, #5cd0ff 100%);
    box-shadow: 0 8px 18px rgba(29, 94, 240, 0.25);
  }

  .brand-mark svg {
    width: 16px;
    height: 16px;
    display: block;
  }

  .brand-copy {
    display: flex;
    flex-direction: column;
  }

  .eyebrow {
    display: none;
  }

  .brand-copy h1 {
    margin: 0;
    font-size: 19px;
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
    padding: 10px;
    margin-bottom: 8px;
  }

  .hero-card h2 {
    margin-bottom: 8px;
    font-size: 21px;
  }

  .hero-card p,
  .hero-card .info-banner {
    display: none;
  }

  .mode-list {
    gap: 8px;
    margin-top: 8px;
  }

  .mode-option {
    min-height: 84px;
    height: 84px;
    padding: 8px 10px;
    width: 100%;
    text-align: left;
    border: 1px solid rgba(18, 38, 63, 0.08);
    background: #ffffff;
    border-radius: 12px;
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
    font-size: 17px;
    display: block;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
  }

  .mode-description {
    font-size: 14px;
    line-height: 1.35;
    display: block;
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
  .secondary-button,
  .pagination-button,
  .text-button {
    min-height: 44px;
    font-size: 15px;
    border: none;
    border-radius: 12px;
    padding: 12px 14px;
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
    padding: 10px;
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
    padding: 10px;
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
    grid-template-columns: repeat(4, minmax(0, 1fr));
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

  .badge.skipped {
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

  .pagination {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
  }

  .pagination-button {
    border: 1px solid rgba(18, 38, 63, 0.12);
    border-radius: 10px;
    background: #ffffff;
    color: #17304c;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }

  .pagination-button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .pagination-count {
    text-align: center;
    font-size: 10px;
    font-weight: 700;
    color: #58708a;
  }


  button, [role="button"] {
    min-height: 44px;
  }

  button:focus-visible, [href]:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid #ffbf47;
    outline-offset: 3px;
  }

  .help-button {
    min-width: 44px;
    min-height: 44px;
    padding: 0 10px;
    font-size: 14px;
    border: 2px solid #17304c;
    border-radius: 999px;
    background: #ffffff;
    color: #17304c;
    font-weight: 800;
    cursor: pointer;
  }

  .mode-option {
    min-height: 84px;
    height: 84px;
    padding: 8px 10px;
  }

  .mode-option-inner {
    display: grid;
    grid-template-columns: 40px 1fr;
    align-items: center;
    gap: 10px;
  }

  .mode-icon {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: #edf4ff;
    color: #17304c;
  }

  .mode-icon svg { width: 24px; height: 24px; }

  .mode-title { font-size: 17px; }
  .mode-description { font-size: 14px; line-height: 1.35; }

  .action-button, .secondary-button, .pagination-button {
    min-height: 44px;
    font-size: 15px;
  }

  .hero-card p, .microcopy, .error-card p, .snippet, .disclaimer,
  .caveat, .pagination-count, .help-copy, .help-list, .help-terms {
    font-size: 16px;
  }

  .eyebrow, .pill, .badge, .confidence, .tag, .summary-stat span {
    font-size: 12px;
  }

  .summary-title h3 { font-size: 18px; }
  .status-copy { font-size: 18px; }

  .help-card {
    background: #ffffff;
    border: 1px solid rgba(18, 38, 63, 0.12);
    border-radius: 16px;
    padding: 14px;
  }

  .help-card h2 {
    margin: 0 0 12px;
    font-size: 24px;
  }

  .help-list {
    margin: 0;
    padding-left: 24px;
  }

  .help-list li {
    margin: 0 0 10px;
    line-height: 1.45;
  }

  .help-terms {
    display: grid;
    gap: 10px;
    margin: 12px 0;
  }

  .help-term {
    padding: 10px;
    border-radius: 10px;
    background: #f2f6fb;
  }

  .help-disclaimer {
    border-left: 4px solid #99621d;
    background: #fff8e8;
    padding: 10px;
    font-size: 16px;
    line-height: 1.45;
  }

  @media (max-width: 399px) {
    html,
    body,
    #root,
    .popup-shell { width: 100%; min-width: 0; max-width: 100%; }
    .popup-shell { padding: 12px; }
    .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .tutorial-actions,
    .consent-actions { grid-template-columns: 1fr; }
  }

  .tutorial-card,
  .consent-card {
    background: #ffffff;
    border: 1px solid rgba(18, 38, 63, 0.12);
    border-radius: 16px;
    padding: 16px;
  }

  .tutorial-progress {
    margin: 0 0 8px;
    font-size: 16px;
    font-weight: 800;
    color: #214e9f;
  }

  .tutorial-card h2,
  .consent-card h2 {
    margin: 0 0 10px;
    font-size: 24px;
  }

  .tutorial-copy,
  .consent-card p {
    margin: 0 0 12px;
    font-size: 16px;
    line-height: 1.55;
    color: #405671;
  }

  .tutorial-progress-dots {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin: 14px 0;
  }

  .tutorial-progress-dots span {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: #ccd7e5;
  }

  .tutorial-progress-dots span.active {
    width: 28px;
    background: #1d5ef0;
  }

  .tutorial-actions,
  .consent-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .consent-actions {
    grid-template-columns: 1fr;
  }

  .text-button {
    width: 100%;
    min-height: 44px;
    margin-top: 8px;
    border: none;
    background: transparent;
    color: #214e9f;
    font-size: 16px;
    font-weight: 800;
    text-decoration: underline;
    cursor: pointer;
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      scroll-behavior: auto !important;
      transition-duration: 0.01ms !important;
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
    }
  }

  .compact-layout {
    width: 400px;
  }

  .popup-shell {
    width: 400px;
    min-height: 480px;
    max-height: 520px;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 10px 12px 12px;
  }

  #root {
    max-height: 520px;
  }

  .topbar {
    min-height: 52px;
    margin-bottom: 6px;
  }

  .brand-mark {
    width: 28px;
    height: 28px;
    border-radius: 8px;
  }

  .brand-mark svg {
    width: 16px;
    height: 16px;
  }

  .eyebrow {
    display: none;
  }

  .brand-copy h1 {
    font-size: 19px;
  }

  .help-button {
    min-width: 44px;
    min-height: 44px;
    padding: 0 10px;
    font-size: 14px;
  }

  .hero-card {
    padding: 10px;
    margin-bottom: 8px;
  }

  .hero-card h2 {
    margin-bottom: 8px;
    font-size: 21px;
  }

  .hero-card p,
  .hero-card .info-banner {
    display: none;
  }

  .mode-list {
    gap: 8px;
    margin-top: 8px;
  }

  .mode-option {
    min-height: 84px;
    height: 84px;
    padding: 8px 10px;
  }

  .mode-option-inner {
    grid-template-columns: 40px 1fr;
    gap: 10px;
  }

  .mode-icon {
    width: 40px;
    height: 40px;
  }

  .mode-title {
    font-size: 17px;
  }

  .mode-description {
    font-size: 14px;
    line-height: 1.35;
  }

  .action-button,
  .secondary-button,
  .pagination-button,
  .text-button {
    min-height: 44px;
    font-size: 15px;
  }

  .tutorial-card,
  .consent-card,
  .help-card {
    max-height: 458px;
    overflow-y: auto;
    padding: 12px;
  }

  .tutorial-progress {
    margin-bottom: 4px;
    font-size: 14px;
  }

  .tutorial-card h2,
  .consent-card h2,
  .help-card h2 {
    margin-bottom: 6px;
    font-size: 21px;
  }

  .tutorial-copy,
  .consent-card p,
  .help-list,
  .help-disclaimer {
    font-size: 14px;
    line-height: 1.4;
  }

  .tutorial-progress-dots {
    margin: 8px 0;
  }

  .tutorial-actions,
  .consent-actions {
    gap: 8px;
  }

  .text-button {
    margin-top: 4px;
  }

  .summary-card,
  .result-card,
  .error-card,
  .scanning-card {
    padding: 10px;
  }
`;

export default function Popup() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [scanMode, setScanMode] = useState<ScanMode>("screen");
  const [resultPage, setResultPage] = useState(0);
  const [helpOpen, setHelpOpen] = useState(false);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [tutorialStep, setTutorialStep] = useState(0);
  const [consentOpen, setConsentOpen] = useState(false);
  const [pendingScanMode, setPendingScanMode] = useState<ScanMode>("screen");
  const [onboarding, setOnboarding] = useState<OnboardingSettings>({
    onboardingCompleted: true,
    uploadNoticeAcknowledged: true,
  });
  const activeScanRef = useRef(0);
  const helpOriginRef = useRef<Status>(status);
  const viewHeadingRef = useRef<HTMLHeadingElement>(null);

  const orderedResults = useMemo(() => {
    if (status.kind !== "done") return [];
    return sortRealEyeItems(status.results, status.skipped);
  }, [status]);

  const paginatedResults = useMemo(() =>
    paginateRealEyeItems(orderedResults, resultPage),
  [orderedResults, resultPage]);

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
      skipped: status.skipped.length,
    };

    return {
      ...counts,
      hasWarnings: counts.warning > 0,
    };
  }, [status]);

  useEffect(() => {
    const frame = window.requestAnimationFrame?.(() => viewHeadingRef.current?.focus())
      ?? null;
    return () => {
      if (frame != null) window.cancelAnimationFrame?.(frame);
    };
  }, [status.kind, helpOpen, tutorialOpen, tutorialStep, consentOpen]);

  useEffect(() => {
    if (status.kind !== "done") return;

    const payload = getPageHighlightPayload(orderedResults, status.domMarkers, resultPage);

    void chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (!tab?.id) return;
      return chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: applyRealEyeHighlights,
        args: [payload],
      });
    }).catch(() => {
      // The page may no longer be available; popup results remain usable.
    });
  }, [resultPage, status, orderedResults]);

  useEffect(() => {
    let cancelled = false;
    void readOnboardingSettings().then((settings) => {
      if (cancelled) return;
      setOnboarding(settings);
      if (shouldShowTutorial(settings)) {
        setTutorialStep(0);
        setTutorialOpen(true);
      }
    });
    return () => { cancelled = true; };
  }, []);

  function openHelp() {
    helpOriginRef.current = status;
    setHelpOpen(true);
  }

  function closeHelp() {
    setHelpOpen(false);
    setStatus(helpOriginRef.current);
  }

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
    setResultPage(0);
    void clearPageHighlights();
    setStatus({ kind: "idle" });
  }

  function closeResults() {
    activeScanRef.current += 1;
    setResultPage(0);
    void clearPageHighlights();
    setStatus({ kind: "idle" });
  }

  async function startScan(requestedMode: ScanMode = scanMode) {
    const scanToken = ++activeScanRef.current;
    setResultPage(0);
    setStatus({ kind: "scanning" });
    await clearPageHighlights();
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id || !tab.url) throw new Error("No active tab found.");

      const extractionFunction = (requestedMode === "screen"
        ? extractVisibleTextBlocks
        : extractSelectedTextBlocks) as (...args: unknown[]) => unknown;
      let extractedBlocks: Array<ExtractedTextBlock | string> = [];
      try {
        const [injection] = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: extractionFunction,
        });
        extractedBlocks = (injection.result ?? []) as Array<ExtractedTextBlock | string>;
      } catch {
        throw new Error("Chrome does not allow RealEye to scan this page.");
      }

      if (!isActiveScanToken(activeScanRef.current, scanToken)) return;

      const blocks = extractedBlocks.map((block) =>
        typeof block === "string"
          ? { id: crypto.randomUUID(), text: block, selector: "" }
          : {
              id: block.id ?? crypto.randomUUID(),
              text: block.text ?? "",
              selector: block.selector ?? "",
              skipped: block.skipped === true,
              skipExplanation: block.skipExplanation,
            },
      );

      if (blocks.length === 0) {
        const noVisibleTextMessage = requestedMode === "screen"
          ? "No eligible visible text was found in the current viewport. Scroll to the article content and try again."
          : "No text was selected on this page. Select webpage text and try again.";
        throw new Error(noVisibleTextMessage);
      }

      const scanId = crypto.randomUUID();
      const textMap = new Map<string, string>();
      const markerMap = new Map<string, string>();
      const allResults: ItemResult[] = [];
      const skipped: SkippedDisplayItem[] = [];
      const validBlocks = blocks.filter((block) => !block.skipped);

      for (const block of blocks) {
        if (!block.skipped) continue;
        const itemId = crypto.randomUUID();
        textMap.set(itemId, block.text);
        skipped.push({
          item_id: itemId,
          domMarkerId: block.id,
          text: block.text,
          explanation: block.skipExplanation ?? "Unable to analyze",
        });
      }

      for (let index = 0; index < validBlocks.length; index += 10) {
        const batch = validBlocks.slice(index, index + 10);
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
        if (!isActiveScanToken(activeScanRef.current, scanToken)) return;
      }

      if (!isActiveScanToken(activeScanRef.current, scanToken)) return;

      setStatus({ kind: "done", results: allResults, skipped, texts: textMap, domMarkers: markerMap });
    } catch (err) {
      if (!isActiveScanToken(activeScanRef.current, scanToken)) return;
      const message = friendlyApiError(err);
      setStatus({ kind: "error", message });
    }
  }

  async function completeTutorial() {
    await setTutorialCompleted();
    setOnboarding((current) => ({ ...current, onboardingCompleted: true }));
    setTutorialOpen(false);
    setTutorialStep(0);
  }

  function skipTutorial() {
    void completeTutorial();
  }

  function replayTutorial() {
    setHelpOpen(false);
    setTutorialStep(0);
    setTutorialOpen(true);
  }

  function cancelUploadConsent() {
    setConsentOpen(false);
    setStatus({ kind: "idle" });
  }

  async function continueUploadConsent() {
    await setUploadNoticeAcknowledged();
    setOnboarding((current) => ({ ...current, uploadNoticeAcknowledged: true }));
    const mode = pendingScanMode;
    setConsentOpen(false);
    await startScan(mode);
  }

  function requestScan(mode: ScanMode) {
    setScanMode(mode);
    if (shouldShowUploadNotice(onboarding)) {
      setPendingScanMode(mode);
      setConsentOpen(true);
      return;
    }
    void startScan(mode);
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
          {!helpOpen && !tutorialOpen && !consentOpen && (
            <button type="button" className="help-button" onClick={openHelp} aria-label="Open RealEye help">
              Help
            </button>
          )}
        </header>

        {tutorialOpen && (
          <section className="tutorial-card" aria-labelledby="realeye-tutorial-title">
            <p className="tutorial-progress" aria-live="polite">Step {tutorialStep + 1} of {TUTORIAL_STEPS.length}</p>
            <h2 id="realeye-tutorial-title" ref={viewHeadingRef} tabIndex={-1}>{TUTORIAL_STEPS[tutorialStep].title}</h2>
            <p className="tutorial-copy">{TUTORIAL_STEPS[tutorialStep].body}</p>
            <div className="tutorial-progress-dots" aria-hidden="true">
              {TUTORIAL_STEPS.map((step, index) => (
                <span key={step.title} className={index === tutorialStep ? "active" : ""} />
              ))}
            </div>
            <div className="tutorial-actions">
              <button type="button" className="secondary-button" disabled={tutorialStep === 0} onClick={() => setTutorialStep((step) => Math.max(0, step - 1))}>Back</button>
              {tutorialStep < TUTORIAL_STEPS.length - 1 ? (
                <button type="button" className="action-button" onClick={() => setTutorialStep((step) => Math.min(TUTORIAL_STEPS.length - 1, step + 1))}>Next</button>
              ) : (
                <button type="button" className="action-button" onClick={() => void completeTutorial()}>Finish</button>
              )}
            </div>
            <button type="button" className="text-button" onClick={skipTutorial}>Skip Tutorial</button>
          </section>
        )}

        {consentOpen && (
          <section className="consent-card" role="dialog" aria-modal="true" aria-labelledby="realeye-consent-title">
            <h2 id="realeye-consent-title" ref={viewHeadingRef} tabIndex={-1}>Before the first scan</h2>
            <p>
              RealEye extracts eligible text from the {pendingScanMode === "screen" ? "currently visible webpage viewport" : "text you selected on the webpage"}.
            </p>
            <p>
              {pendingScanMode === "screen"
                ? "Screen Scan does not collect images. Eligible text is sent to the RealEye backend for analysis."
                : "Selected Content analyzes only the text you selected. That selected text is sent to the RealEye backend for analysis."}
            </p>
            <p>The scan has not started yet.</p>
            <div className="consent-actions">
              <button type="button" className="action-button" onClick={() => void continueUploadConsent()}>Continue with Scan</button>
              <button type="button" className="secondary-button" onClick={cancelUploadConsent}>Cancel</button>
            </div>
          </section>
        )}

        {!tutorialOpen && !consentOpen && helpOpen && (
          <section className="help-card" aria-labelledby="realeye-help-title">
            <h2 id="realeye-help-title" ref={viewHeadingRef} tabIndex={-1}>RealEye Help</h2>
            <ol className="help-list">
              {HELP_STEPS.map((step) => <li key={step}>{step}</li>)}
            </ol>

            <div className="help-terms" aria-label="Result meanings">
              <div className="help-term"><strong>Possible AI:</strong> {RESULT_MEANINGS.possibleAi}</div>
              <div className="help-term"><strong>No AI signs found:</strong> {RESULT_MEANINGS.noAi}</div>
              <div className="help-term"><strong>Inconclusive:</strong> {RESULT_MEANINGS.inconclusive}</div>
              <div className="help-term"><strong>Skipped:</strong> {RESULT_MEANINGS.skipped}</div>
            </div>

            <button type="button" className="action-button" onClick={replayTutorial}>Replay Tutorial</button>
            <button type="button" className="text-button" onClick={closeHelp}>Back</button>
          </section>
        )}

        {!tutorialOpen && !consentOpen && !helpOpen && (
          <>
            <div className="hero-card">
              <h2 ref={viewHeadingRef} tabIndex={-1}>What would you like to scan?</h2>

              <div className="mode-list" role="group" aria-label="Select scan mode">
                <button type="button" className={`mode-option ${scanMode === "screen" ? "selected" : ""}`} onClick={() => setScanMode("screen")} aria-pressed={scanMode === "screen"}>
                  <div className="mode-option-inner">
                    <div className="mode-icon" aria-hidden="true">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="3" y="4" width="18" height="16" rx="2" stroke="currentColor" strokeWidth="2"/><path d="M8 9h8M8 13h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                    </div>
                    <div>
                      <span className="mode-title">{POPUP_MODE_COPY.screen.title}</span>
                      <span className="mode-description">{POPUP_MODE_COPY.screen.description}</span>
                    </div>
                  </div>
                </button>

                <button type="button" className={`mode-option ${scanMode === "selected" ? "selected" : ""}`} onClick={() => setScanMode("selected")} aria-pressed={scanMode === "selected"}>
                  <div className="mode-option-inner">
                    <div className="mode-icon" aria-hidden="true">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M5 5h14v14H5z" stroke="currentColor" strokeWidth="2"/><path d="M8 12h8M12 8v8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                    </div>
                    <div>
                      <span className="mode-title">{POPUP_MODE_COPY.selected.title}</span>
                      <span className="mode-description">{POPUP_MODE_COPY.selected.description}</span>
                    </div>
                  </div>
                </button>
              </div>


              <button type="button" className="action-button" onClick={() => requestScan(scanMode)}>Start scan</button>
            </div>

            {status.kind === "scanning" && (
              <div className="scanning-card">
                <div className="status-row">
                  <div className="spinner" aria-hidden="true" />
                  <p className="status-copy">Scanning in progress...</p>
                </div>
                <p className="microcopy">This may take a few moments. Please don't close the popup or navigate away.</p>
              </div>
            )}

            {status.kind === "error" && (
              <div className="error-card">
                <div className="error-title">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2.75C8.98 2.75 6.5 4.71 6.5 7.67V9.9c0 3.62 2.07 6.83 5.2 8.64a1.3 1.3 0 0 0 1.2 0c3.13-1.81 5.2-5.02 5.2-8.64V7.67c0-2.96-2.48-4.92-5.5-4.92Zm-1.1 8.2 1.41 1.41 3.09-3.1 1.1 1.1-4.2 4.2-2.5-2.5 1.1-1.1Z" fill="currentColor"/>
                  </svg>
                  <span>Scan error</span>
                </div>
                <p className="microcopy">{status.message}</p>
                <button type="button" className="action-button" onClick={cancelScan}>Try again</button>
              </div>
            )}

            {status.kind === "done" && (
              <div className="results-wrap">
                <section className="summary-card">
                  <div className="summary-title">
                    <span className={`summary-dot ${summary?.hasWarnings ? "warning" : ""}`} aria-hidden="true" />
                    <h3 ref={viewHeadingRef} tabIndex={-1}>{summary?.hasWarnings ? "Possible AI content detected" : "No strong signal detected"}</h3>
                  </div>
                  <div className="summary-grid">
                    <div className="summary-stat"><strong>{status.results.length + status.skipped.length}</strong><span>Items</span></div>
                    <div className="summary-stat"><strong>{summary?.warning ?? 0}</strong><span>Possible AI</span></div>
                    <div className="summary-stat"><strong>{summary?.inconclusive ?? 0}</strong><span>No AI signs</span></div>
                    <div className="summary-stat"><strong>{summary?.skipped ?? 0}</strong><span>Skipped</span></div>
                  </div>
                </section>

                <ul className="result-list">
                  {paginatedResults.items.map((entry) => {
                    if (entry.kind === "skipped") {
                      const preview = status.texts.get(entry.item.item_id)?.trim().replace(/\s+/g, " ") ?? "";
                      return (
                        <li key={entry.item.item_id} className="result-card">
                          <div className="result-top"><span className="badge skipped">Skipped</span></div>
                          {preview && <p className="snippet">“{preview.slice(0, 120)}”</p>}
                          <div className="meta"><span className="tag neutral">{entry.item.explanation}</span></div>
                        </li>
                      );
                    }

                    const result = entry.result;
                    const preview = status.texts.get(result.item_id)?.trim().replace(/\s+/g, " ") ?? "";
                    const estimate = result.classifier.estimate != null ? Math.round(result.classifier.estimate * 100) : null;
                    const signalSummary = result.warning_evidence.length > 0
                      ? result.warning_evidence.map((signal) => EVIDENCE_LABELS[signal] ?? signal).join(" · ")
                      : "No signal triggered";

                    return (
                      <li key={result.item_id} className="result-card">
                        <div className="result-top">
                          <span className={`badge ${result.overall_state}`}>{STATE_LABELS[result.overall_state]}</span>
                          {estimate != null && <span className="confidence">Model estimate: {estimate}%</span>}
                        </div>
                        {preview && <p className="snippet">“{preview.slice(0, 120)}”</p>}
                        {estimate != null && (
                          <div className={`meter ${result.overall_state === "warning" ? "warning" : ""}`}><span style={{ width: `${Math.min(100, Math.max(8, estimate))}%` }} /></div>
                        )}
                        <div className="meta"><span className={`tag ${result.overall_state === "warning" ? "warning" : "neutral"}`}>{signalSummary}</span></div>
                        {result.overall_state !== "unable_to_analyze" && <p className="caveat">Estimate only. May be inaccurate.</p>}
                      </li>
                    );
                  })}
                </ul>

                {orderedResults.length > 15 && (
                  <nav className="pagination" aria-label="Result pages">
                    <button type="button" className="pagination-button" disabled={resultPage === 0} onClick={() => setResultPage((page) => Math.max(0, page - 1))}>Previous</button>
                    <span className="pagination-count">Showing {paginatedResults.start + 1}–{paginatedResults.end} of {orderedResults.length}</span>
                    <button type="button" className="pagination-button" disabled={resultPage >= paginatedResults.totalPages - 1} onClick={() => setResultPage((page) => Math.min(paginatedResults.totalPages - 1, page + 1))}>Next</button>
                  </nav>
                )}

                <div className="disclaimer">AI detection can be wrong. Missing credentials or watermarks do not prove that a person created this content.</div>
                <button type="button" className="secondary-button" onClick={closeResults}>Close</button>
              </div>
            )}
          </>
        )}
      </main>
    </>
  );
}
