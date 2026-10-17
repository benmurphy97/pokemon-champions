import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' lets the built site be served from any folder (e.g. GitHub Pages).
// The operating room polls /api/ops, which champions.ops_server answers (python -m champions.ops_server).
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
})
