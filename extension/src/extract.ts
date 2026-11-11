// Runs INSIDE the webpage (injected with chrome.scripting.executeScript),
// so it must be self-contained: no imports, no outside variables.
// Text-only for now; images come later.
export interface ExtractedTextBlock {
  id: string;
  text: string;
  selector: string;
}

export interface VisibleTextExtractionDiagnostics {
  noCandidates: number;
  viewportExcluded: number;
  chromeExcluded: number;
  lengthExcluded: number;
  deduplicated: number;
  accepted: number;
}

export function clearRealEyeHighlights(): void {
  document.querySelectorAll('[data-realeye-overlay="true"]').forEach((node) => node.remove());
}

export function applyRealEyeHighlights(
  highlights: Array<{
    domMarkerId?: string;
    item_id: string;
    overall_state: string;
    text?: string;
    explanation?: string;
  }>,
): void {
  const clearLocalHighlights = (): void => {
    document.querySelectorAll('[data-realeye-overlay="true"]').forEach((node) => node.remove());
  };

  clearLocalHighlights();
  const matches: Array<{ node: Element; highlight: typeof highlights[number] }> = [];

  document.querySelectorAll('[data-realeye-element-id]').forEach((node) => {
    const markerId = node.getAttribute("data-realeye-element-id");
    if (!markerId) return;

    const match = highlights.find((item) => item.domMarkerId === markerId || item.item_id === markerId);
    if (match) matches.push({ node, highlight: match });
  });

  matches.forEach(({ node, highlight }) => {
    if (highlight.overall_state !== "warning") return;

    const rect = node.getBoundingClientRect();
    if (!rect) return;

    const overlay = document.createElement("div");
    overlay.setAttribute("data-realeye-overlay", "true");
    overlay.style.position = "fixed";
    overlay.style.left = `${rect.left + window.scrollX}px`;
    overlay.style.top = `${rect.top + window.scrollY}px`;
    overlay.style.width = `${Math.max(rect.width, 40)}px`;
    overlay.style.height = `${Math.max(rect.height, 28)}px`;
    overlay.style.pointerEvents = "auto";
    overlay.style.border = "2px solid rgba(214, 109, 34, 0.95)";
    overlay.style.borderRadius = "8px";
    overlay.style.background = "rgba(255, 181, 89, 0.08)";
    overlay.style.boxShadow = "0 10px 24px rgba(214, 109, 34, 0.18)";
    overlay.style.zIndex = "2147483647";

    const label = document.createElement("div");
    label.textContent = "Possible AI";
    label.style.position = "absolute";
    label.style.left = "6px";
    label.style.top = "-20px";
    label.style.padding = "4px 8px";
    label.style.borderRadius = "999px";
    label.style.background = "#d76d22";
    label.style.color = "#ffffff";
    label.style.fontSize = "10px";
    label.style.fontWeight = "700";
    label.style.letterSpacing = "0.04em";
    label.style.textTransform = "uppercase";
    label.style.whiteSpace = "nowrap";

    const actions = document.createElement("div");
    actions.style.position = "absolute";
    actions.style.left = "6px";
    actions.style.top = "calc(100% + 6px)";
    actions.style.display = "flex";
    actions.style.gap = "6px";
    actions.style.flexWrap = "wrap";

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.textContent = "Open explanation";
    openButton.setAttribute("aria-label", "Open explanation for this flagged text");
    openButton.style.background = "#ffffff";
    openButton.style.color = "#17304c";
    openButton.style.border = "1px solid rgba(23, 48, 76, 0.18)";
    openButton.style.padding = "4px 8px";
    openButton.style.borderRadius = "999px";
    openButton.style.fontSize = "10px";
    openButton.style.fontWeight = "700";
    openButton.style.cursor = "pointer";
    openButton.addEventListener("click", () => {
      const explanation = document.createElement("div");
      explanation.style.position = "fixed";
      explanation.style.left = "12px";
      explanation.style.bottom = "12px";
      explanation.style.width = "260px";
      explanation.style.maxWidth = "calc(100vw - 24px)";
      explanation.style.background = "#ffffff";
      explanation.style.border = "1px solid rgba(23, 48, 76, 0.16)";
      explanation.style.borderRadius = "12px";
      explanation.style.boxShadow = "0 18px 30px rgba(15, 33, 52, 0.15)";
      explanation.style.padding = "12px";
      explanation.style.zIndex = "2147483648";
      explanation.style.color = "#17304c";
      explanation.style.fontSize = "12px";
      explanation.style.lineHeight = "1.5";
      explanation.style.fontFamily = "Inter, Segoe UI, sans-serif";
      explanation.innerHTML = `<div style="font-weight:700; margin-bottom:6px;">AI likelihood warning</div><div>${highlight.explanation ?? "This content matches the existing warning state returned by the model. The popup is using the backend warning state as the signal until final threshold calibration is published."}</div>`;
      const dismissBox = document.createElement("button");
      dismissBox.type = "button";
      dismissBox.textContent = "Dismiss";
      dismissBox.style.marginTop = "10px";
      dismissBox.style.background = "#edf3fb";
      dismissBox.style.border = "1px solid rgba(23, 48, 76, 0.12)";
      dismissBox.style.borderRadius = "999px";
      dismissBox.style.padding = "6px 10px";
      dismissBox.style.fontSize = "11px";
      dismissBox.style.fontWeight = "700";
      dismissBox.style.cursor = "pointer";
      dismissBox.addEventListener("click", () => explanation.remove());
      explanation.appendChild(dismissBox);
      document.body.appendChild(explanation);
    });

    const dismissButton = document.createElement("button");
    dismissButton.type = "button";
    dismissButton.textContent = "Dismiss";
    dismissButton.setAttribute("aria-label", "Dismiss this highlight");
    dismissButton.style.background = "#ffffff";
    dismissButton.style.color = "#17304c";
    dismissButton.style.border = "1px solid rgba(23, 48, 76, 0.18)";
    dismissButton.style.padding = "4px 8px";
    dismissButton.style.borderRadius = "999px";
    dismissButton.style.fontSize = "10px";
    dismissButton.style.fontWeight = "700";
    dismissButton.style.cursor = "pointer";
    dismissButton.addEventListener("click", () => overlay.remove());

    actions.appendChild(openButton);
    actions.appendChild(dismissButton);
    overlay.appendChild(label);
    overlay.appendChild(actions);
    document.body.appendChild(overlay);
  });
}

function buildSelector(element: Element): string {
  if (element.id) return `#${CSS.escape ? CSS.escape(element.id) : element.id}`;

  const tag = element.tagName.toLowerCase();
  const className = element.getAttribute("class");
  if (className && className.trim()) {
    const classes = className.split(/\s+/).filter(Boolean).slice(0, 3).map((value) => `.${value}`).join("");
    return `${tag}${classes}`;
  }

  return tag;
}

function rectsIntersectViewport(rects: ArrayLike<DOMRect>): boolean {
  if (!rects || typeof rects.length !== "number") return false;

  for (let index = 0; index < rects.length; index += 1) {
    const rect = rects[index];
    if (!rect) continue;

    const visible =
      rect.width > 0 &&
      rect.height > 0 &&
      rect.bottom > 0 &&
      rect.top < window.innerHeight &&
      rect.right > 0 &&
      rect.left < window.innerWidth;

    if (visible) return true;
  }

  return false;
}

function intersectsViewport(element: Element): boolean {
  if (!(element instanceof HTMLElement)) return false;

  if (element.getClientRects && rectsIntersectViewport(element.getClientRects())) {
    return true;
  }

  const rect = element.getBoundingClientRect();
  if (
    rect &&
    rect.width > 0 &&
    rect.height > 0 &&
    rect.bottom > 0 &&
    rect.top < window.innerHeight &&
    rect.right > 0 &&
    rect.left < window.innerWidth
  ) {
    return true;
  }

  return Array.from(element.children).some((child) => intersectsViewport(child));
}

function isHiddenElement(element: Element): boolean {
  let current: Element | null = element;

  while (current) {
    if (!(current instanceof HTMLElement)) {
      current = current.parentElement;
      continue;
    }

    const style = window.getComputedStyle(current);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.opacity === "0" ||
      style.clip === "rect(0, 0, 0, 0)" ||
      style.fontSize === "0px" ||
      current.getAttribute("aria-hidden") === "true"
    ) {
      return true;
    }

    current = current.parentElement;
  }

  return false;
}

export function getVisibleTextExtractionDiagnostics(): VisibleTextExtractionDiagnostics {
  const normalizeText = (value: string): string =>
    value.replace(/\s+/g, " ").replace(/\u00a0/g, " ").trim();

  const rectsIntersectViewport = (rects: ArrayLike<DOMRect>): boolean => {
    if (!rects || typeof rects.length !== "number") return false;

    for (let index = 0; index < rects.length; index += 1) {
      const rect = rects[index];
      if (!rect) continue;

      const visible =
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom > 0 &&
        rect.top < window.innerHeight &&
        rect.right > 0 &&
        rect.left < window.innerWidth;

      if (visible) return true;
    }

    return false;
  };

  const intersectsViewport = (element: Element): boolean => {
    if (!(element instanceof HTMLElement)) return false;

    if (element.getClientRects && rectsIntersectViewport(element.getClientRects())) {
      return true;
    }

    const rect = element.getBoundingClientRect();
    if (
      rect &&
      rect.width > 0 &&
      rect.height > 0 &&
      rect.bottom > 0 &&
      rect.top < window.innerHeight &&
      rect.right > 0 &&
      rect.left < window.innerWidth
    ) {
      return true;
    }

    return Array.from(element.children).some((child) => intersectsViewport(child));
  };

  const isHiddenElement = (element: Element): boolean => {
    let current: Element | null = element;

    while (current) {
      if (!(current instanceof HTMLElement)) {
        current = current.parentElement;
        continue;
      }

      const style = window.getComputedStyle(current);
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        style.opacity === "0" ||
        style.clip === "rect(0, 0, 0, 0)" ||
        style.fontSize === "0px" ||
        current.getAttribute("aria-hidden") === "true"
      ) {
        return true;
      }

      current = current.parentElement;
    }

    return false;
  };

  const hasExcludedAncestor = (element: Element): boolean => {
    const excludedSelector = [
      "nav",
      "header",
      "footer",
      "aside",
      "form",
      "button",
      "[role='navigation']",
      "[role='banner']",
      "[role='contentinfo']",
      ".nav",
      ".navbar",
      ".menu",
      ".sidebar",
      ".breadcrumb",
      ".footer",
      ".header",
      ".ads",
      ".cookie",
      ".share",
      ".social",
    ].join(", ");

    return element.closest(excludedSelector) !== null;
  };

  const isCandidateElement = (element: Element): boolean => {
    if (
      element.matches(
        "script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed",
      )
    ) {
      return false;
    }

    if (hasExcludedAncestor(element) || isHiddenElement(element)) {
      return false;
    }

    return true;
  };

  const getCandidateRoots = (): Element[] => {
    const preferredSelectors = [
      "article",
      "main",
      "[role='main']",
      ".article",
      ".entry-content",
      ".post-content",
      ".content",
      ".story",
      "section",
    ];

    const preferred = Array.from(document.querySelectorAll(preferredSelectors.join(", ")));
    if (preferred.length > 0) {
      return preferred;
    }

    return document.body ? [document.body] : [];
  };

  const MIN_CHARS = 200;
  const MAX_CHARS = 5000;

  const diagnostics: VisibleTextExtractionDiagnostics = {
    noCandidates: 0,
    viewportExcluded: 0,
    chromeExcluded: 0,
    lengthExcluded: 0,
    deduplicated: 0,
    accepted: 0,
  };

  const candidateRoots = getCandidateRoots();
  const selector = "article, main, [role='main'], section, p, li, blockquote";
  const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const fallback = Array.from(document.querySelectorAll("p, li, blockquote")).filter(
    (node) => !candidates.includes(node),
  );

  if (candidates.length === 0 && fallback.length === 0) {
    diagnostics.noCandidates = 1;
    return diagnostics;
  }

  const acceptedNodes: Element[] = [];
  const allCandidates = [...candidates, ...fallback];

  for (const candidate of allCandidates) {
    if (!isCandidateElement(candidate)) {
      diagnostics.chromeExcluded += 1;
      continue;
    }

    if (!intersectsViewport(candidate)) {
      diagnostics.viewportExcluded += 1;
      continue;
    }

    const text = normalizeText(candidate.textContent ?? "");
    if (text.length < MIN_CHARS || text.length > MAX_CHARS) {
      diagnostics.lengthExcluded += 1;
      continue;
    }

    const ancestorMatch = acceptedNodes.find((existing) => existing.contains(candidate));
    if (ancestorMatch) {
      diagnostics.deduplicated += 1;
      continue;
    }

    const childMatchIndex = acceptedNodes.findIndex((existing) => candidate.contains(existing));
    if (childMatchIndex >= 0) {
      acceptedNodes.splice(childMatchIndex, 1);
      diagnostics.deduplicated += 1;
    }

    acceptedNodes.push(candidate);
    diagnostics.accepted += 1;
  }

  return diagnostics;
}

export function extractVisibleTextBlocks(): ExtractedTextBlock[] {
  const normalizeText = (value: string): string =>
    value.replace(/\s+/g, " ").replace(/\u00a0/g, " ").trim();

  const buildSelector = (element: Element): string => {
    if (element.id) return `#${CSS.escape ? CSS.escape(element.id) : element.id}`;

    const tag = element.tagName.toLowerCase();
    const className = element.getAttribute("class");
    if (className && className.trim()) {
      const classes = className.split(/\s+/).filter(Boolean).slice(0, 3).map((value) => `.${value}`).join("");
      return `${tag}${classes}`;
    }

    return tag;
  };

  const rectsIntersectViewport = (rects: ArrayLike<DOMRect>): boolean => {
    if (!rects || typeof rects.length !== "number") return false;

    for (let index = 0; index < rects.length; index += 1) {
      const rect = rects[index];
      if (!rect) continue;

      const visible =
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom > 0 &&
        rect.top < window.innerHeight &&
        rect.right > 0 &&
        rect.left < window.innerWidth;

      if (visible) return true;
    }

    return false;
  };

  const intersectsViewport = (element: Element): boolean => {
    if (!(element instanceof HTMLElement)) return false;

    if (element.getClientRects && rectsIntersectViewport(element.getClientRects())) {
      return true;
    }

    const rect = element.getBoundingClientRect();
    if (
      rect &&
      rect.width > 0 &&
      rect.height > 0 &&
      rect.bottom > 0 &&
      rect.top < window.innerHeight &&
      rect.right > 0 &&
      rect.left < window.innerWidth
    ) {
      return true;
    }

    return Array.from(element.children).some((child) => intersectsViewport(child));
  };

  const isHiddenElement = (element: Element): boolean => {
    let current: Element | null = element;

    while (current) {
      if (!(current instanceof HTMLElement)) {
        current = current.parentElement;
        continue;
      }

      const style = window.getComputedStyle(current);
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        style.opacity === "0" ||
        style.clip === "rect(0, 0, 0, 0)" ||
        style.fontSize === "0px" ||
        current.getAttribute("aria-hidden") === "true"
      ) {
        return true;
      }

      current = current.parentElement;
    }

    return false;
  };

  const hasExcludedAncestor = (element: Element): boolean => {
    const excludedSelector = [
      "nav",
      "header",
      "footer",
      "aside",
      "form",
      "button",
      "[role='navigation']",
      "[role='banner']",
      "[role='contentinfo']",
      ".nav",
      ".navbar",
      ".menu",
      ".sidebar",
      ".breadcrumb",
      ".footer",
      ".header",
      ".ads",
      ".cookie",
      ".share",
      ".social",
    ].join(", ");

    return element.closest(excludedSelector) !== null;
  };

  const isCandidateElement = (element: Element): boolean => {
    if (
      element.matches(
        "script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed",
      )
    ) {
      return false;
    }

    if (hasExcludedAncestor(element) || isHiddenElement(element)) {
      return false;
    }

    return true;
  };

  const getCandidateRoots = (): Element[] => {
    const preferredSelectors = [
      "article",
      "main",
      "[role='main']",
      ".article",
      ".entry-content",
      ".post-content",
      ".content",
      ".story",
      "section",
    ];

    const preferred = Array.from(document.querySelectorAll(preferredSelectors.join(", ")));
    if (preferred.length > 0) {
      return preferred;
    }

    return document.body ? [document.body] : [];
  };

  const getDiagnostics = (): VisibleTextExtractionDiagnostics => {
    const diagnostics: VisibleTextExtractionDiagnostics = {
      noCandidates: 0,
      viewportExcluded: 0,
      chromeExcluded: 0,
      lengthExcluded: 0,
      deduplicated: 0,
      accepted: 0,
    };

    const candidateRoots = getCandidateRoots();
    const selector = "article, main, [role='main'], section, p, li, blockquote";
    const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
    const fallback = Array.from(document.querySelectorAll("p, li, blockquote")).filter(
      (node) => !candidates.includes(node),
    );

    if (candidates.length === 0 && fallback.length === 0) {
      diagnostics.noCandidates = 1;
      return diagnostics;
    }

    const acceptedNodes: Element[] = [];
    const allCandidates = [...candidates, ...fallback];

    for (const candidate of allCandidates) {
      if (!isCandidateElement(candidate)) {
        diagnostics.chromeExcluded += 1;
        continue;
      }

      if (!intersectsViewport(candidate)) {
        diagnostics.viewportExcluded += 1;
        continue;
      }

      const text = normalizeText(candidate.textContent ?? "");
      if (text.length < 200 || text.length > 5000) {
        diagnostics.lengthExcluded += 1;
        continue;
      }

      const ancestorMatch = acceptedNodes.find((existing) => existing.contains(candidate));
      if (ancestorMatch) {
        diagnostics.deduplicated += 1;
        continue;
      }

      const childMatchIndex = acceptedNodes.findIndex((existing) => candidate.contains(existing));
      if (childMatchIndex >= 0) {
        acceptedNodes.splice(childMatchIndex, 1);
        diagnostics.deduplicated += 1;
      }

      acceptedNodes.push(candidate);
      diagnostics.accepted += 1;
    }

    return diagnostics;
  };

  const MIN_CHARS = 200;
  const MAX_CHARS = 5000;
  const MAX_BLOCKS = 10;

  const candidateRoots = getCandidateRoots();
  const selector = "article, main, [role='main'], section, p, li, blockquote";
  const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const fallback = Array.from(document.querySelectorAll("p, li, blockquote")).filter(
    (node) => !candidates.includes(node),
  );

  document.querySelectorAll('[data-realeye-element-id]').forEach((node) => {
    node.removeAttribute("data-realeye-element-id");
  });

  const acceptedNodes: Element[] = [];

  for (const candidate of [...candidates, ...fallback]) {
    if (!isCandidateElement(candidate)) continue;
    if (!intersectsViewport(candidate)) continue;

    const text = normalizeText(candidate.textContent ?? "");
    if (text.length < MIN_CHARS || text.length > MAX_CHARS) continue;

    const ancestorMatch = acceptedNodes.find((existing) => existing.contains(candidate));
    if (ancestorMatch) continue;

    const childMatchIndex = acceptedNodes.findIndex((existing) => candidate.contains(existing));
    if (childMatchIndex >= 0) {
      acceptedNodes.splice(childMatchIndex, 1);
    }

    candidate.setAttribute("data-realeye-element-id", `realeye-${Math.random().toString(36).slice(2, 10)}`);
    acceptedNodes.push(candidate);
    if (acceptedNodes.length >= MAX_BLOCKS) break;
  }

  const diagnostics = getDiagnostics();
  if ((globalThis as typeof globalThis & { __REAL_EYE_DEBUG__?: boolean }).__REAL_EYE_DEBUG__ === true) {
    console.debug("[RealEye viewport diagnostics]", diagnostics);
  }

  return acceptedNodes
    .map((candidate) => ({
      id: candidate.getAttribute("data-realeye-element-id") ?? `realeye-${Math.random().toString(36).slice(2, 10)}`,
      text: normalizeText(candidate.textContent ?? ""),
      selector: buildSelector(candidate),
    }))
    .filter((block) => block.text.length >= MIN_CHARS && block.text.length <= MAX_CHARS);
}

export function extractTextBlocks(): string[] {
  const normalizeText = (value: string): string =>
    value.replace(/\s+/g, " ").replace(/\u00a0/g, " ").trim();

  const hasExcludedAncestor = (element: Element): boolean => {
    const excludedSelector = [
      "nav",
      "header",
      "footer",
      "aside",
      "form",
      "button",
      "[role='navigation']",
      "[role='banner']",
      "[role='contentinfo']",
      ".nav",
      ".navbar",
      ".menu",
      ".sidebar",
      ".breadcrumb",
      ".footer",
      ".header",
      ".ads",
      ".cookie",
      ".share",
      ".social",
    ].join(", ");

    return element.closest(excludedSelector) !== null;
  };

  const isHiddenElement = (element: Element): boolean => {
    if (!(element instanceof HTMLElement)) return false;

    const style = window.getComputedStyle(element);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.opacity === "0" ||
      style.clip === "rect(0, 0, 0, 0)" ||
      style.fontSize === "0px"
    ) {
      return true;
    }

    return element.getAttribute("aria-hidden") === "true";
  };

  const isCandidateElement = (element: Element): boolean => {
    if (
      element.matches(
        "script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed",
      )
    ) {
      return false;
    }

    if (hasExcludedAncestor(element) || isHiddenElement(element)) {
      return false;
    }

    return true;
  };

  const getCandidateRoots = (): Element[] => {
    const preferredSelectors = [
      "article",
      "main",
      "[role='main']",
      ".article",
      ".entry-content",
      ".post-content",
      ".content",
      ".story",
      "section",
    ];

    const preferred = Array.from(document.querySelectorAll(preferredSelectors.join(", ")));
    if (preferred.length > 0) {
      return preferred;
    }

    return document.body ? [document.body] : [];
  };

  const MIN_CHARS = 200;
  const MAX_CHARS = 5000;
  const MAX_BLOCKS = 10;

  const candidateRoots = getCandidateRoots();
  const selector = "article, main, [role='main'], section, p, li, blockquote";
  const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const fallback = Array.from(document.querySelectorAll("p, li, blockquote")).filter(
    (node) => !candidates.includes(node),
  );

  const acceptedNodes: Element[] = [];

  for (const candidate of [...candidates, ...fallback]) {
    if (!isCandidateElement(candidate)) continue;

    const text = normalizeText(candidate.textContent ?? "");
    if (text.length < MIN_CHARS || text.length > MAX_CHARS) continue;

    const ancestorMatch = acceptedNodes.find((existing) => existing.contains(candidate));
    if (ancestorMatch) continue;

    const childMatchIndex = acceptedNodes.findIndex((existing) => candidate.contains(existing));
    if (childMatchIndex >= 0) {
      acceptedNodes.splice(childMatchIndex, 1);
    }

    acceptedNodes.push(candidate);
    if (acceptedNodes.length >= MAX_BLOCKS) break;
  }

  return acceptedNodes
    .map((candidate) => normalizeText(candidate.textContent ?? ""))
    .filter(Boolean);
}
