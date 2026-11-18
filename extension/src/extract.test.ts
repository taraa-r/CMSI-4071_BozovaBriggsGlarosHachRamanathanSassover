/// <reference types="node" />
import assert from "node:assert/strict";
import test from "node:test";

import {
  applyRealEyeHighlights,
  extractTextBlocks,
  extractVisibleTextBlocks,
  extractSelectedTextBlocks,
  getVisibleTextExtractionDiagnostics,
  sortRealEyeItems,
  paginateRealEyeItems,
  getPageHighlightPayload,
  isActiveScanToken,
} from "./extract.ts";

type StyleMap = {
  display?: string;
  visibility?: string;
  opacity?: string;
  clip?: string;
  fontSize?: string;
  outline?: string;
  boxShadow?: string;
};

class FakeElement {
  tagName: string;
  parent: FakeElement | null;
  children: FakeElement[];
  attributes: Map<string, string>;
  private _textContent: string;
  style: StyleMap;
  rect: { top: number; left: number; width: number; height: number; right: number; bottom: number };
  private listeners: Map<string, Array<(event: unknown) => void>> = new Map();

  constructor(tagName: string, parent: FakeElement | null = null, textContent = "") {
    this.tagName = tagName.toLowerCase();
    this.parent = parent;
    this.children = [];
    this.attributes = new Map();
    this._textContent = textContent;
    this.style = {};
    this.rect = { top: 0, left: 0, width: 100, height: 100, right: 100, bottom: 100 };
  }

  get textContent(): string {
    const childText = this.children.map((child) => child.textContent).join("");
    return `${this._textContent ?? ""}${childText}`;
  }

  set textContent(value: string) {
    this._textContent = value;
  }

  appendChild(child: FakeElement): FakeElement {
    child.parent = this;
    this.children.push(child);
    return child;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }

  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  addEventListener(eventName: string, callback: EventListener): void {
    const callbacks = this.listeners.get(eventName) ?? [];
    callbacks.push(callback as unknown as (event: unknown) => void);
    this.listeners.set(eventName, callbacks);
  }

  click(): void {
    for (const callback of this.listeners.get("click") ?? []) callback({});
  }

  remove(): void {
    if (this.parent) {
      this.parent.children = this.parent.children.filter((child) => child !== this);
      this.parent = null;
    }
  }

  matches(selector: string): boolean {
    const selectors = selector.split(",").map((part) => part.trim()).filter(Boolean);

    return selectors.some((part) => {
      if (part === "") return false;

      if (part.startsWith(".")) {
        return this.classList.includes(part.slice(1));
      }

      if (part.startsWith("[")) {
        const attrText = part.slice(1, -1);
        const [name, rawValue] = attrText.split("=");
        if (!name) return false;
        const value = rawValue?.replace(/^['\"]|['\"]$/g, "");
        const attr = this.getAttribute(name);
        return value ? attr === value : attr !== null;
      }

      return this.tagName === part.toLowerCase();
    });
  }

  closest(selector: string): FakeElement | null {
    let current: FakeElement | null = this;
    while (current) {
      if (current.matches(selector)) return current;
      current = current.parent;
    }
    return null;
  }

  contains(node: FakeElement | null): boolean {
    if (!node) return false;
    if (node === this) return true;

    const queue: FakeElement[] = [...this.children];
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (current === node) return true;
      queue.push(...current.children);
    }

    return false;
  }

  querySelectorAll(selector: string): FakeElement[] {
    const results: FakeElement[] = [];

    const walk = (element: FakeElement): void => {
      if (element.matches(selector)) {
        results.push(element);
      }
      for (const child of element.children) {
        walk(child);
      }
    };

    walk(this);
    return results;
  }

  querySelector(selector: string): FakeElement | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  getBoundingClientRect() {
    return this.rect;
  }

  get classList(): string[] {
    const classAttr = this.getAttribute("class") ?? "";
    return classAttr.split(/\s+/).filter(Boolean);
  }

  get parentElement(): FakeElement | null {
    return this.parent;
  }

  getClientRects(): FakeElement["rect"][] {
    return [this.rect];
  }
}

class FakeDocument {
  body: FakeElement;

  constructor(body: FakeElement) {
    this.body = body;
  }

  createElement(tagName: string): FakeElement {
    return new FakeElement(tagName);
  }

  querySelectorAll(selector: string): FakeElement[] {
    return this.body.querySelectorAll(selector);
  }

  querySelector(selector: string): FakeElement | null {
    return this.body.querySelector(selector);
  }
}

function setGlobalDom(root: FakeElement) {
  const document = new FakeDocument(root);
  const HTMLElement = FakeElement;

  Object.assign(globalThis, {
    document,
    HTMLElement,
    CSS: { escape: (value: string) => value },
    window: {
      innerWidth: 800,
      innerHeight: 600,
      scrollX: 0,
      scrollY: 0,
      getComputedStyle: (element: FakeElement) => element.style,
    },
  });
}

test("extractTextBlocks prefers readable article content and ignores navigation and hidden text", () => {
  const body = new FakeElement("body");
  const nav = new FakeElement("nav");
  nav.textContent = "Home About Contact Blog";
  body.appendChild(nav);

  const main = new FakeElement("main");
  const article = new FakeElement("article");
  const paragraph = new FakeElement("p");
  paragraph.textContent =
    "This article describes a detailed product review with enough substantive context to exceed the minimum threshold for analysis. The passage discusses design, performance, and customer expectations in a realistic way so it can be assessed for authenticity and signal quality.";
  article.appendChild(paragraph);
  main.appendChild(article);
  body.appendChild(main);

  const footer = new FakeElement("footer");
  footer.textContent = "Privacy policy Terms of service";
  body.appendChild(footer);

  const hidden = new FakeElement("p");
  hidden.style.visibility = "hidden";
  hidden.textContent = "This hidden text is not eligible for scanning because it is visually hidden.";
  body.appendChild(hidden);

  setGlobalDom(body);

  const blocks = extractTextBlocks();

  assert.equal(blocks.length, 1);
  assert.ok(blocks[0].text.includes("This article describes a detailed product review"));
  assert.ok(!blocks[0].text.includes("Home About Contact"));
});

test("extractTextBlocks remains executable after serialization", () => {
  const body = new FakeElement("body");
  const article = new FakeElement("article");
  const paragraph = new FakeElement("p");
  paragraph.textContent =
    "This article is long enough to be sent to the RealEye backend for analysis and is intentionally written to match the minimum valid payload length required by the project contract. It should still be available after serialization because the function is self-contained when injected into the webpage context.";
  article.appendChild(paragraph);
  body.appendChild(article);

  setGlobalDom(body);

  const serializedFunction = extractTextBlocks.toString();
  const injected = new Function(`return (${serializedFunction})`)();
  const blocks = injected();

  assert.equal(blocks.length, 1);
  assert.ok(blocks[0].text.includes("This article is long enough"));
});

test("injected extraction and highlight functions remain self-contained after serialization", () => {
  const body = new FakeElement("body");
  const article = new FakeElement("article");
  const paragraph = new FakeElement("p");
  paragraph.textContent =
    "This paragraph is long enough to meet the RealEye viewport threshold and is intentionally written so the injected screen scan function can be serialized without any closure dependencies. It deliberately contains enough meaningful content to remain above the minimum text threshold while preserving the same page-scoped extraction logic that the browser injects.";
  article.appendChild(paragraph);
  body.appendChild(article);

  setGlobalDom(body);

  const visibleScan = new Function(`return (${extractVisibleTextBlocks.toString()})`)();
  const blocks = visibleScan();
  assert.equal(blocks.length, 1);

  const highlight = new Function(`return (${applyRealEyeHighlights.toString()})`)();
  assert.doesNotThrow(() =>
    highlight([{ item_id: "test", domMarkerId: blocks[0].id, overall_state: "warning", explanation: "Test warning" }]),
  );
});

test("extractTextBlocks represents text below the minimum length requirement as skipped", () => {
  const body = new FakeElement("body");
  const paragraph = new FakeElement("p");
  paragraph.textContent = "Too short to scan.";
  body.appendChild(paragraph);

  setGlobalDom(body);

  const blocks = extractTextBlocks();
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0].skipped, true);
  assert.equal(blocks[0].skipReason, "too_short");
  assert.equal(blocks[0].skipExplanation, "Too short to analyze");
});

test("extractTextBlocks deduplicates nested article content", () => {
  const body = new FakeElement("body");
  const article = new FakeElement("article");
  const intro = new FakeElement("p");
  intro.textContent =
    "This is a long substantive paragraph repeated across nested elements so the extraction should keep only one clean content block to avoid sending duplicate text to the backend. This sentence ensures the text stays above the required length threshold for the RealEye API.";
  const wrapper = new FakeElement("section");
  wrapper.appendChild(intro);
  article.appendChild(wrapper);
  body.appendChild(article);

  setGlobalDom(body);

  const blocks = extractTextBlocks();
  assert.equal(blocks.length, 1);
});

test("extractVisibleTextBlocks filters to viewport content and tags the source element", () => {
  const body = new FakeElement("body");
  const article = new FakeElement("article");
  article.rect = { top: 80, left: 20, width: 500, height: 200, right: 520, bottom: 280 };
  const visibleParagraph = new FakeElement("p");
  visibleParagraph.rect = { top: 120, left: 30, width: 440, height: 90, right: 470, bottom: 210 };
  visibleParagraph.textContent =
    "This content is visible within the current browser viewport and long enough to qualify for a meaningful RealEye scan. It should be included in the screen scan because it intersects the active viewing area and not hidden navigation content.";
  const outsideParagraph = new FakeElement("p");
  outsideParagraph.rect = { top: 700, left: 10, width: 300, height: 80, right: 310, bottom: 780 };
  outsideParagraph.textContent =
    "This content is off-screen and should be excluded from the viewport-only extraction because it sits below the currently visible area and does not intersect the user’s viewport.";

  article.appendChild(visibleParagraph);
  article.appendChild(outsideParagraph);
  body.appendChild(article);

  setGlobalDom(body);

  const blocks = extractVisibleTextBlocks();
  assert.equal(blocks.length, 1);
  assert.ok(blocks[0].text.includes("visible within the current browser viewport"));
  assert.equal(document.body.querySelectorAll('[data-realeye-element-id]').length, 1);
  const markerId = document.body.querySelector('[data-realeye-element-id]')?.getAttribute('data-realeye-element-id');
  assert.ok(markerId);
});

test("extractVisibleTextBlocks keeps visible text when the parent box is zero-sized but a child is visible", () => {
  const body = new FakeElement("body");
  const article = new FakeElement("article");
  const wrapper = new FakeElement("p");
  wrapper.style.display = "contents";
  wrapper.rect = { top: 0, left: 0, width: 0, height: 0, right: 0, bottom: 0 };

  const inlineText = new FakeElement("span");
  inlineText.rect = { top: 150, left: 40, width: 440, height: 70, right: 480, bottom: 220 };
  inlineText.textContent =
    "This visible article text is rendered by a child span, so the zero-sized parent wrapper must not cause the whole block to be rejected. The viewport detector should still accept the paragraph because its visible descendant intersects the screen.";

  wrapper.appendChild(inlineText);
  article.appendChild(wrapper);
  body.appendChild(article);

  setGlobalDom(body);

  const diagnostics = getVisibleTextExtractionDiagnostics();
  assert.equal(diagnostics.noCandidates, 0);
  assert.equal(diagnostics.viewportExcluded, 0);

  const blocks = extractVisibleTextBlocks();
  assert.equal(blocks.length, 1);
  assert.ok(blocks[0].text.includes("This visible article text is rendered by a child span"));
});

test("sortRealEyeItems puts warnings first, human results next, and skipped items last", () => {
  const classified = [
    { item_id: "human", overall_state: "inconclusive" as const },
    { item_id: "warning", overall_state: "warning" as const },
  ];
  const skipped = [{
    item_id: "skipped",
    domMarkerId: "realeye-short",
    text: "Short",
    explanation: "Too short to analyze",
  }];

  const ordered = sortRealEyeItems(classified, skipped);
  assert.deepEqual(ordered.map((entry) =>
    entry.kind === "skipped" ? entry.item.item_id : entry.result.item_id
  ), ["warning", "human", "skipped"]);
});

test("extractTextBlocks includes blockquotes and figcaptions without duplicate nested paragraphs", () => {
  const body = new FakeElement("body");
  const article = new FakeElement("article");
  const quote = new FakeElement("blockquote");
  const quoteParagraph = new FakeElement("p");
  quoteParagraph.textContent = "A clearly attributed quotation that contains enough substantive wording to qualify as an individual text block. It is nested inside a blockquote and should not be extracted twice.";
  quote.appendChild(quoteParagraph);

  const figure = new FakeElement("figure");
  const caption = new FakeElement("figcaption");
  caption.textContent = "This image caption is also eligible text and provides meaningful context for readers while remaining separate from the surrounding article paragraph content.";
  figure.appendChild(caption);

  article.appendChild(quote);
  article.appendChild(figure);
  body.appendChild(article);
  setGlobalDom(body);

  const blocks = extractTextBlocks();
  assert.equal(blocks.filter((block) => block.text.includes("clearly attributed quotation")).length, 1);
  assert.equal(blocks.filter((block) => block.text.includes("image caption is also eligible")).length, 1);
});

test("extractTextBlocks represents excessively long blocks as skipped", () => {
  const body = new FakeElement("body");
  const paragraph = new FakeElement("p");
  paragraph.textContent = "A".repeat(5001);
  body.appendChild(paragraph);
  setGlobalDom(body);

  const blocks = extractTextBlocks();
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0].skipped, true);
  assert.equal(blocks[0].skipReason, "too_long");
  assert.equal(blocks[0].skipExplanation, "Too long to analyze");
});

test("paginateRealEyeItems shows 15-result pages and preserves range metadata", () => {
  const items = Array.from({ length: 42 }, (_, index) => index);
  assert.equal(paginateRealEyeItems(items, 0).items.length, 15);
  assert.equal(paginateRealEyeItems(items, 0).start, 0);
  assert.equal(paginateRealEyeItems(items, 1).start, 15);
  assert.equal(paginateRealEyeItems(items, 2).end, 42);
  assert.equal(paginateRealEyeItems(items, 99).end, 42);
});

test("getPageHighlightPayload only includes warning results on the current page", () => {
  const ordered = [
    { kind: "classified" as const, result: { item_id: "w1", overall_state: "warning" as const, explanation: "Actual warning explanation" } },
    { kind: "classified" as const, result: { item_id: "h1", overall_state: "inconclusive" as const } },
    { kind: "skipped" as const, item: { item_id: "s1", domMarkerId: "m-s1", text: "Short", explanation: "Too short" } },
  ];
  const payload = getPageHighlightPayload(ordered, new Map([["w1", "m-w1"]]), 0);
  assert.deepEqual(payload, [{ item_id: "w1", domMarkerId: "m-w1", overall_state: "warning", explanation: "Actual warning explanation" }]);
});

test("isActiveScanToken rejects stale scans after cancellation", () => {
  assert.equal(isActiveScanToken(2, 2), true);
  assert.equal(isActiveScanToken(3, 2), false);
});

test("applyRealEyeHighlights creates only warning overlays and a real backend explanation", () => {
  const body = new FakeElement("body");
  const paragraph = new FakeElement("p");
  paragraph.setAttribute("data-realeye-element-id", "marker-one");
  body.appendChild(paragraph);
  setGlobalDom(body);

  applyRealEyeHighlights([{ item_id: "one", domMarkerId: "marker-one", overall_state: "warning", explanation: "Backend explanation text" }]);

  assert.equal(body.querySelectorAll('[data-realeye-overlay="true"]').length, 1);
  assert.equal(paragraph.style.outline, "3px solid #d76d22");

  const openButton = body.querySelectorAll("button").find((button) => button.textContent === "Open Explanation");
  if (!openButton) throw new Error("Open Explanation button was not created");
  openButton.click();

  const explanationPanel = body.querySelector('[data-realeye-explanation="true"]');
  if (!explanationPanel) throw new Error("Explanation panel was not created");
  assert.ok(explanationPanel.textContent.includes("Backend explanation text"));
});

test("individual warning dismissal removes only its overlay and border", () => {
  const body = new FakeElement("body");
  const first = new FakeElement("p");
  const second = new FakeElement("p");
  first.setAttribute("data-realeye-element-id", "marker-one");
  second.setAttribute("data-realeye-element-id", "marker-two");
  body.appendChild(first);
  body.appendChild(second);
  setGlobalDom(body);

  applyRealEyeHighlights([
    { item_id: "one", domMarkerId: "marker-one", overall_state: "warning", explanation: "First" },
    { item_id: "two", domMarkerId: "marker-two", overall_state: "warning", explanation: "Second" },
  ]);

  const dismissButtons = body.querySelectorAll("button").filter((button) => button.textContent === "Dismiss");
  assert.equal(dismissButtons.length, 2);
  dismissButtons[0].click();

  assert.equal(body.querySelectorAll('[data-realeye-overlay="true"]').length, 1);
  assert.equal(first.style.outline, "");
  assert.equal(second.style.outline, "3px solid #d76d22");
});

test("applying highlights repeatedly does not duplicate warning overlays", () => {
  const body = new FakeElement("body");
  const paragraph = new FakeElement("p");
  paragraph.setAttribute("data-realeye-element-id", "marker-one");
  body.appendChild(paragraph);
  setGlobalDom(body);

  const payload = [{ item_id: "one", domMarkerId: "marker-one", overall_state: "warning", explanation: "Repeated" }];
  applyRealEyeHighlights(payload);
  applyRealEyeHighlights(payload);

  assert.equal(body.querySelectorAll('[data-realeye-overlay="true"]').length, 1);
  assert.equal(body.querySelectorAll('[data-realeye-element-id]').length, 1);
});

test("extractSelectedTextBlocks returns only the active selected text", () => {
  const body = new FakeElement("body");
  const paragraph = new FakeElement("p");
  paragraph.textContent = "This paragraph contains selected text and unrelated surrounding text. The selected text itself must be the payload.";
  body.appendChild(paragraph);
  setGlobalDom(body);

  Object.assign(globalThis.window, {
    getSelection: () => ({
      toString: () => "The selected text itself must be the payload.",
      anchorNode: { parentElement: paragraph },
    }),
  });

  const blocks = extractSelectedTextBlocks();
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0].text, "The selected text itself must be the payload.");
  assert.equal(blocks[0].skipped, true);
  assert.equal(blocks[0].skipReason, "too_short");
});
