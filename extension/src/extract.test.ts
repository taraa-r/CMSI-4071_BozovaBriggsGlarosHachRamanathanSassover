import assert from "node:assert/strict";
import test from "node:test";

import { extractTextBlocks } from "./extract.ts";

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
  textContent: string;
  style: StyleMap;

  constructor(tagName: string, parent: FakeElement | null = null, textContent = "") {
    this.tagName = tagName.toLowerCase();
    this.parent = parent;
    this.children = [];
    this.attributes = new Map();
    this.textContent = textContent;
    this.style = {};
  }

  appendChild(child: FakeElement): FakeElement {
    child.parent = this;
    this.children.push(child);
    return child;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
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

  querySelectorAll(selector: string): FakeElement[] {
    return this.body.querySelectorAll(selector);
  }
}

function setGlobalDom(root: FakeElement) {
  const document = new FakeDocument(root);
  const HTMLElement = FakeElement;

  Object.assign(globalThis, {
    document,
    HTMLElement,
    window: {
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
