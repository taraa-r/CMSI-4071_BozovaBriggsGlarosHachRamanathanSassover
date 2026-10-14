import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Builds the popup into dist/. Load dist/ in chrome://extensions as an unpacked extension.
export default defineConfig({
  plugins: [react()],
});
