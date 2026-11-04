// Runs INSIDE the webpage (injected with chrome.scripting.executeScript),
// so it must be self-contained: no imports, no outside variables.
// Text-only for now; images come later.
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

  const MIN_CHARS = 200; // SRS 5.5.3
  const MAX_CHARS = 5000;
  const MAX_BLOCKS = 10; // one /analyze request; batching comes later

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
