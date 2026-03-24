import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['core/tests/**/*.test.ts'],
  },
});
