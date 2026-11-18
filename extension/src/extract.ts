// Runs INSIDE the webpage (injected with chrome.scripting.executeScript),
// so it must be self-contained: no imports, no outside variables.
// Text-only for now; images come later.
export interface ExtractedTextBlock {
  id: string;
  text: string;
  selector: string;
  skipped?: boolean;
  skipReason?: "too_short" | "too_long";
  skipExplanation?: string;
}

export interface VisibleTextExtractionDiagnostics {
  noCandidates: number;
  viewportExcluded: number;
  chromeExcluded: number;
  lengthExcluded: number;
  deduplicated: number;
  accepted: number;
}

export interface SortableClassificationResult {
  item_id: string;
  overall_state: "warning" | "inconclusive" | "unable_to_analyze";
}

export interface SkippedDisplayItem {
  item_id: string;
  domMarkerId: string;
  text: string;
  explanation: string;
}

export function sortRealEyeItems<T extends SortableClassificationResult>(
  classified: T[],
  skipped: SkippedDisplayItem[],
): Array<{ kind: "classified"; result: T } | { kind: "skipped"; item: SkippedDisplayItem }> {
  const warning = classified.filter((result) => result.overall_state === "warning");
  const human = classified.filter((result) => result.overall_state === "inconclusive");
  const unable = classified.filter((result) => result.overall_state === "unable_to_analyze");

  return [
    ...warning.map((result) => ({ kind: "classified" as const, result })),
    ...human.map((result) => ({ kind: "classified" as const, result })),
    ...unable.map((result) => ({ kind: "classified" as const, result })),
    ...skipped.map((item) => ({ kind: "skipped" as const, item })),
  ];
}

export function extractSelectedTextBlocks(): ExtractedTextBlock[] {
  const normalizeText = (value: string): string =>
    value.replace(/\s+/g, " ").replace(/\u00a0/g, " ").trim();

  const selection = window.getSelection ? window.getSelection() : null;
  const text = normalizeText(selection?.toString() ?? "");
  if (!text) return [];

  let target: Element | null = null;
  const anchorNode = selection?.anchorNode ?? null;
  if (anchorNode) {
    const anchorElement = anchorNode instanceof HTMLElement
      ? anchorNode
      : anchorNode.parentElement;
    target = anchorElement?.closest("p, li, blockquote, figcaption, article, main, section, div") ?? anchorElement;
  }

  if (!target) target = document.body;
  if (!target || (target as HTMLElement).matches?.("nav, header, footer, aside, form, button")) return [];

  const id = `realeye-${Math.random().toString(36).slice(2, 10)}`;
  target.setAttribute("data-realeye-element-id", id);

  const block: ExtractedTextBlock = {
    id,
    text,
    selector: target.id
      ? `#${CSS.escape ? CSS.escape(target.id) : target.id}`
      : target.tagName.toLowerCase(),
  };

  if (text.length < 200) {
    block.skipped = true;
    block.skipReason = "too_short";
    block.skipExplanation = "Too short to analyze";
  } else if (text.length > 5000) {
    block.skipped = true;
    block.skipReason = "too_long";
    block.skipExplanation = "Too long to analyze";
  }

  return [block];
}

export function clearRealEyeHighlights(): void {
  document.querySelectorAll('[data-realeye-overlay="true"], [data-realeye-explanation="true"]').forEach((node) => node.remove());
  document.querySelectorAll('[data-realeye-element-id]').forEach((node) => {
    if (!(node instanceof HTMLElement)) return;
    const originalOutline = node.getAttribute("data-realeye-original-outline") ?? "";
    const originalBoxShadow = node.getAttribute("data-realeye-original-box-shadow") ?? "";
    node.style.outline = originalOutline;
    node.style.boxShadow = originalBoxShadow;
  });
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
    document.querySelectorAll('[data-realeye-overlay="true"], [data-realeye-explanation="true"]').forEach((node) => node.remove());
    document.querySelectorAll('[data-realeye-element-id]').forEach((node) => {
      if (!(node instanceof HTMLElement)) return;
      const originalOutline = node.getAttribute("data-realeye-original-outline") ?? "";
      const originalBoxShadow = node.getAttribute("data-realeye-original-box-shadow") ?? "";
      node.style.outline = originalOutline;
      node.style.boxShadow = originalBoxShadow;
    });
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

    if (node instanceof HTMLElement) {
      node.setAttribute("data-realeye-original-outline", node.style.outline || "");
      node.setAttribute("data-realeye-original-box-shadow", node.style.boxShadow || "");
      node.style.outline = "3px solid #d76d22";
      node.style.outlineOffset = "2px";
      node.style.boxShadow = `${node.style.boxShadow ? `${node.style.boxShadow}, ` : ""}0 0 0 6px rgba(243, 163, 58, 0.18)`;
    }

    const overlay = document.createElement("div");
    overlay.setAttribute("data-realeye-overlay", "true");
    overlay.style.position = "fixed";
    overlay.style.left = `${Math.max(0, rect.left)}px`;
    overlay.style.top = `${Math.max(0, rect.top)}px`;
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
    openButton.textContent = "Open Explanation";
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
      explanation.setAttribute("data-realeye-explanation", "true");
      explanation.setAttribute("role", "dialog");
      explanation.setAttribute("aria-label", "Possible AI explanation");
      const heading = document.createElement("div");
      heading.textContent = "AI likelihood warning";
      heading.style.fontWeight = "700";
      heading.style.marginBottom = "6px";
      const body = document.createElement("div");
      body.textContent = typeof highlight.explanation === "string" && highlight.explanation.trim()
        ? highlight.explanation
        : "No explanation was provided by the analysis service.";
      explanation.appendChild(heading);
      explanation.appendChild(body);
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
    dismissButton.addEventListener("click", () => {
      overlay.remove();
      document.querySelectorAll(`[data-realeye-explanation="true"]`).forEach((panel) => panel.remove());
      if (node instanceof HTMLElement) {
        node.style.outline = node.getAttribute("data-realeye-original-outline") ?? "";
        node.style.outlineOffset = "";
        node.style.boxShadow = node.getAttribute("data-realeye-original-box-shadow") ?? "";
      }
    });

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

    return !hasExcludedAncestor(element) && !isHiddenElement(element);
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
  const selector = "p, li, blockquote, figcaption";
  const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const fallback = Array.from(document.querySelectorAll("p, li, blockquote, figcaption")).filter(
    (node) => !candidates.includes(node),
  );

  if (candidates.length === 0 && fallback.length === 0) {
    diagnostics.noCandidates = 1;
    return diagnostics;
  }

  const acceptedNodes: Element[] = [];
  const allCandidates = [...candidates, ...fallback];

  for (const candidate of allCandidates) {
    if (
      candidate.matches("script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed") ||
      hasExcludedAncestor(candidate) ||
      isHiddenElement(candidate)
    ) {
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
    const selector = "p, li, blockquote, figcaption";
    const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
    const fallback = Array.from(document.querySelectorAll("p, li, blockquote, figcaption")).filter(
      (node) => !candidates.includes(node),
    );

    if (candidates.length === 0 && fallback.length === 0) {
      diagnostics.noCandidates = 1;
      return diagnostics;
    }

    const acceptedNodes: Element[] = [];
    const allCandidates = [...candidates, ...fallback];

    for (const candidate of allCandidates) {
      if (
        candidate.matches("script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed") ||
        hasExcludedAncestor(candidate) ||
        isHiddenElement(candidate)
      ) {
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
  const selector = "p, li, blockquote, figcaption";
  const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const fallback = Array.from(document.querySelectorAll("p, li, blockquote, figcaption")).filter(
    (node) => !candidates.includes(node),
  );

  document.querySelectorAll('[data-realeye-element-id]').forEach((node) => {
    node.removeAttribute("data-realeye-element-id");
  });

  const blocks: ExtractedTextBlock[] = [];

  for (const candidate of [...candidates, ...fallback]) {
    if (candidate.matches("script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed") || hasExcludedAncestor(candidate) || isHiddenElement(candidate)) continue;
    if (!intersectsViewport(candidate)) continue;

    const text = normalizeText(candidate.textContent ?? "");
    if (!text) continue;

    const ancestorMatch = blocks.find((existing) => {
      const existingNode = document.querySelector(`[data-realeye-element-id="${existing.id}"]`);
      return existingNode?.contains(candidate) === true;
    });
    if (ancestorMatch) continue;

    const markerId = `realeye-${Math.random().toString(36).slice(2, 10)}`;
    candidate.setAttribute("data-realeye-element-id", markerId);

    if (text.length < MIN_CHARS) {
      blocks.push({
        id: markerId,
        text,
        selector: buildSelector(candidate),
        skipped: true,
        skipReason: "too_short",
        skipExplanation: "Too short to analyze",
      });
      continue;
    }

    if (text.length > MAX_CHARS) {
      blocks.push({
        id: markerId,
        text,
        selector: buildSelector(candidate),
        skipped: true,
        skipReason: "too_long",
        skipExplanation: "Too long to analyze",
      });
      continue;
    }

    blocks.push({
      id: markerId,
      text,
      selector: buildSelector(candidate),
    });
  }

  const diagnostics = getDiagnostics();
  if ((globalThis as typeof globalThis & { __REAL_EYE_DEBUG__?: boolean }).__REAL_EYE_DEBUG__ === true) {
    console.debug("[RealEye viewport diagnostics]", diagnostics);
  }

  return blocks;
}

export function extractTextBlocks(): ExtractedTextBlock[] {
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
      ) return true;
      current = current.parentElement;
    }
    return false;
  };

  const isCandidateElement = (element: Element): boolean => {
    if (
      element.matches(
        "script, style, noscript, svg, iframe, textarea, input, select, option, img, picture, video, audio, canvas, object, embed",
      )
    ) return false;

    return !hasExcludedAncestor(element) && !isHiddenElement(element);
  };

  const getCandidateRoots = (): Element[] => {
    const preferredSelectors = [
      "article", "main", "[role='main']", ".article", ".entry-content",
      ".post-content", ".content", ".story", "section",
    ];
    const preferred = Array.from(document.querySelectorAll(preferredSelectors.join(", ")));
    return preferred.length > 0 ? preferred : (document.body ? [document.body] : []);
  };

  const MIN_CHARS = 200;
  const MAX_CHARS = 5000;
  const candidateRoots = getCandidateRoots();
  const selector = "p, li, blockquote, figcaption";
  const candidates = candidateRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const fallback = Array.from(document.querySelectorAll("p, li, blockquote, figcaption")).filter(
    (node) => !candidates.includes(node),
  );

  const acceptedNodes: Element[] = [];

  for (const candidate of [...candidates, ...fallback]) {
    if (!isCandidateElement(candidate)) continue;

    const text = normalizeText(candidate.textContent ?? "");
    if (!text) continue;

    const ancestorMatch = acceptedNodes.find((existing) => existing.contains(candidate));
    if (ancestorMatch) continue;

    acceptedNodes.push(candidate);
  }

  return acceptedNodes.map((candidate) => {
    const text = normalizeText(candidate.textContent ?? "");
    const id = `realeye-${Math.random().toString(36).slice(2, 10)}`;
    candidate.setAttribute("data-realeye-element-id", id);

    const block: ExtractedTextBlock = {
      id,
      text,
      selector: candidate.id
        ? `#${CSS.escape ? CSS.escape(candidate.id) : candidate.id}`
        : candidate.tagName.toLowerCase(),
    };

    if (text.length < MIN_CHARS) {
      block.skipped = true;
      block.skipReason = "too_short";
      block.skipExplanation = "Too short to analyze";
    } else if (text.length > MAX_CHARS) {
      block.skipped = true;
      block.skipReason = "too_long";
      block.skipExplanation = "Too long to analyze";
    }

    return block;
  });
}

export const RESULT_PAGE_SIZE = 15;

export function paginateRealEyeItems<T>(items: T[], page: number): { items: T[]; start: number; end: number; totalPages: number } {
  const safePage = Number.isFinite(page) ? Math.max(0, Math.floor(page)) : 0;
  const totalPages = Math.max(1, Math.ceil(items.length / RESULT_PAGE_SIZE));
  const currentPage = Math.min(safePage, totalPages - 1);
  const start = currentPage * RESULT_PAGE_SIZE;
  const end = Math.min(start + RESULT_PAGE_SIZE, items.length);
  return { items: items.slice(start, end), start, end, totalPages };
}

export function getPageHighlightPayload(
  orderedEntries: Array<{ kind: "classified"; result: SortableClassificationResult & { explanation?: string } } | { kind: "skipped"; item: SkippedDisplayItem }>,
  domMarkers: Map<string, string>,
  page: number,
): Array<{ item_id: string; domMarkerId: string; overall_state: string; explanation: string }> {
  const pageItems = paginateRealEyeItems(orderedEntries, page).items;
  return pageItems
    .filter((entry): entry is { kind: "classified"; result: SortableClassificationResult & { explanation?: string } } => entry.kind === "classified")
    .filter((entry) => entry.result.overall_state === "warning")
    .map((entry) => ({
      item_id: entry.result.item_id,
      domMarkerId: domMarkers.get(entry.result.item_id) ?? entry.result.item_id,
      overall_state: entry.result.overall_state,
      explanation: entry.result.explanation ?? "",
    }));
}

export function isActiveScanToken(activeToken: number, scanToken: number): boolean {
  return activeToken === scanToken;
}
