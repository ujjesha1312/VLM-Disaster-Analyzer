import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // Load .env.development / .env.production so the proxy target
  // stays in sync with VITE_API_URL instead of being hardcoded.
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/predict': { target: backendUrl, changeOrigin: true },
        '/models':  { target: backendUrl, changeOrigin: true },
        '/chat':    { target: backendUrl, changeOrigin: true },
      },
    },
  }
})
