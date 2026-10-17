import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' lets the built site be served from any folder (e.g. GitHub Pages).
export default defineConfig({ plugins: [react()], base: './' })
