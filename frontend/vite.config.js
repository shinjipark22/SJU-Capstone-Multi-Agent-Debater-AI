import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 백엔드 경로들. dev 서버가 그대로 FastAPI 로 넘겨주므로 프론트는 상대 경로로만 호출한다
// (LAN IP / SSH 포트포워딩 / 터널 어디로 접속하든 API 주소를 따로 맞출 필요가 없다).
const API_PATHS = ['/debate', '/topics', '/health', '/evaluation', '/sessions', '/survey', '/surveys']
const API_TARGET = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8001'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [
        path,
        // SSE(text/event-stream) 를 버퍼링 없이 흘려보내야 해서 압축을 끈다.
        { target: API_TARGET, changeOrigin: true, ws: false, headers: { 'Accept-Encoding': 'identity' } },
      ]),
    ),
  },
})
