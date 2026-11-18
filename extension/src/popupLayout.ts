export const POPUP_LAYOUT = {
  width: 400,
  minHeight: 480,
  maxHeight: 520,
  narrowViewport: 399,
} as const;

export function isPopupWidthSupported(width: number): boolean {
  return width >= 200 && width <= 800;
}
