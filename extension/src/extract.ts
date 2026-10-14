// Runs INSIDE the webpage (injected with chrome.scripting.executeScript),
// so it must be self-contained: no imports, no outside variables.
// Text-only for now; images come later.
export function extractTextBlocks(): string[] {
  const MIN_CHARS = 200; // SRS 5.5.3
  const MAX_CHARS = 5000;
  const MAX_BLOCKS = 10; // one /analyze request; batching comes later

  const seen = new Set<string>();
  const blocks: string[] = [];

  for (const el of document.querySelectorAll("p, li, blockquote")) {
    const text = (el as HTMLElement).innerText.replace(/\s+/g, " ").trim();
    if (text.length < MIN_CHARS || text.length > MAX_CHARS) continue;
    if (seen.has(text)) continue;
    seen.add(text);
    blocks.push(text);
    if (blocks.length === MAX_BLOCKS) break;
  }
  return blocks;
}
