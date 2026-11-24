import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative base so the build works under any sub-path (GitHub Pages, Netlify, Vercel...).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false },
});
