import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var proxyTarget = env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";
    return {
        plugins: [react()],
        server: {
            host: "127.0.0.1",
            port: 5173,
            proxy: {
                "/api": {
                    target: proxyTarget,
                    changeOrigin: true,
                    rewrite: function (path) { return path.replace(/^\/api/, ""); }
                }
            }
        }
    };
});
