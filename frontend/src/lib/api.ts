/**
 * Thin typed client for the SignLearn backend.
 * Backend lives at SIGNLEARN_BACKEND_URL (default http://127.0.0.1:5001).
 */

const BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL) ||
  "http://127.0.0.1:5001";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${body || path}`);
  }
  return (await res.json()) as T;
}

export interface HealthResponse {
  status: string;
  uptime_seconds?: number;
  model_sha?: string;
  backend?: string;
  num_classes?: number;
}

export interface MetricsResponse {
  predictions_total?: number;
  no_hand_frames_total?: number;
  prediction_latency_ms_p50?: number;
  prediction_latency_ms_p95?: number;
  predictions_per_minute?: number;
  [key: string]: unknown;
}

export interface TranscriptEntry {
  id?: string;
  source: "sign" | "speech";
  text: string;
  confidence?: number;
  ts?: number;
  speaker?: string;
}

export const api = {
  health: () => http<HealthResponse>("/health"),
  metrics: () => http<MetricsResponse>("/metrics?format=json"),
  transcript: (roomId?: string) =>
    http<{ entries: TranscriptEntry[] }>(
      `/transcript${roomId ? `?room_id=${encodeURIComponent(roomId)}` : ""}`,
    ),
  feedback: (payload: { category: string; message: string; context?: string }) =>
    http<{ ok: true }>("/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  correction: (payload: {
    predicted: string;
    actual: string;
    confidence?: number;
    timestamp?: number;
  }) =>
    http<{ ok: true }>("/corrections", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export const BACKEND_URL = BASE;
