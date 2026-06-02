import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy is bypassed when API_BASE_URL uses an absolute URL (e.g. ngrok).
    // Re-enable and update target when running against a local backend.
    proxy: {
      '/predict': { target: 'https://providing-earthy-phonebook.ngrok-free.dev', changeOrigin: true },
      '/models':  { target: 'https://providing-earthy-phonebook.ngrok-free.dev', changeOrigin: true },
      '/chat':    { target: 'https://providing-earthy-phonebook.ngrok-free.dev', changeOrigin: true },
    },
  },
})
