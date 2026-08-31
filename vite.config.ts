/// <reference types="vitest/config" />
import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    rollupOptions: {
      input: {
        game: 'index.html',
        replay: 'replay.html',
      },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'snowgym/tests/**/*.test.ts'],
    globals: true,
  },
});
