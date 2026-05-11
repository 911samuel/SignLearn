import type { NextConfig } from "next";

const config: NextConfig = {
  // Prevent bundling MediaPipe for the server — it uses browser globals at import time.
  serverExternalPackages: ["@mediapipe/tasks-vision"],
  eslint: {
    // ESLint is run separately in CI; don't block production builds.
    ignoreDuringBuilds: true,
  },
};

export default config;
