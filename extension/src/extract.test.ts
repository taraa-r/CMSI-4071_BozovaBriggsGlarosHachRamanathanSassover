import assert from "node:assert/strict";
import test from "node:test";

import {
  applyRealEyeHighlights,
  extractTextBlocks,
  extractVisibleTextBlocks,
  getVisibleTextExtractionDiagnostics,
} from "./extract.ts";

type StyleMap = {
  display?: string;
  visibility?: string;
  opacity?: string;
  clip?: string;
  fontSize?: string;
};

class FakeElement {
  tagName: string;
  parent: FakeElement | null;
  children: FakeElement[];
  attributes: Map<string, string>;
  private _textContent: string;
  style: StyleMap;
  rect: { top: number; left: number; width: number; height: number; right: number; bottom: number };

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

  addEventListener(_eventName: string, _callback: EventListener): void {
    // No-op for the lightweight serialization test harness.
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
  assert.ok(blocks[0].includes("This article describes a detailed product review"));
  assert.ok(!blocks[0].includes("Home About Contact"));
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
  assert.ok(blocks[0].includes("This article is long enough"));
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

test("extractTextBlocks rejects text below the minimum length requirement", () => {
  const body = new FakeElement("body");
  const paragraph = new FakeElement("p");
  paragraph.textContent = "Too short to scan.";
  body.appendChild(paragraph);

  setGlobalDom(body);

  const blocks = extractTextBlocks();
  assert.deepEqual(blocks, []);
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
