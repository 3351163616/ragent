import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
const devProxyTarget = process.env.VITE_DEV_PROXY_TARGET || "http://localhost:9090";
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src")
        }
    },
    server: {
        port: 5173,
        proxy: {
            "/api": {
                target: devProxyTarget,
                changeOrigin: true,
                secure: false
            }
        }
    }
});
