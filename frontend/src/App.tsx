import { useState, useCallback, useEffect } from "react";
import { SignerPanel } from "./components/SignerPanel";
import { HearingPanel } from "./components/HearingPanel";
import { ConversationLog, type LogEntry } from "./components/ConversationLog";
import { type ConnectionStatus } from "./hooks/useSignRecognition";
import "./App.css";

export default function App() {
  const [log, setLog] = useState<LogEntry[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [modelReady, setModelReady] = useState<boolean | null>(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:5001";

  // Check model status on mount
  useEffect(() => {
    fetch(`${BACKEND_URL}/health`)
      .then((r) => r.json())
      .then((data) => setModelReady(data.model_loaded === true))
      .catch(() => setModelReady(false));
  }, [BACKEND_URL]);

  // Hydrate conversation log from server on mount
  useEffect(() => {
    fetch(`${BACKEND_URL}/transcript?limit=200`)
      .then((r) => r.json())
      .then((data) => {
        const hydrated: LogEntry[] = (
          data.messages as Array<{
            id: number;
            ts: string;
            source: "sign" | "speech";
            text: string;
            confidence: number | null;
          }>
        ).map((m) => ({
          id: m.id,
          source: m.source,
          text: m.text,
          confidence: m.confidence ?? undefined,
          ts: new Date(m.ts).getTime(),
        }));
        setLog(hydrated);
      })
      .catch(() => {}); // server not running yet — start with empty log
  }, [BACKEND_URL]);

  const handlePrediction = useCallback((label: string, confidence: number, ts: number) => {
    setLog((prev) => [
      ...prev,
      { id: ts + Math.random(), source: "sign", text: label, confidence, ts },
    ]);
  }, []);

  const handleSpeech = useCallback((text: string, ts: number) => {
    setLog((prev) => [
      ...prev,
      { id: ts + Math.random(), source: "speech", text, ts },
    ]);
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-logo" aria-hidden="true">🤟</span>
        <h1>SignLearn</h1>
      </header>

      {connectionStatus !== "connected" && (
        <div className={`app-banner app-banner--${connectionStatus}`} role="status">
          {connectionStatus === "reconnecting"
            ? "Reconnecting to server…"
            : "Server unavailable — predictions paused. Start the backend and refresh."}
        </div>
      )}
      {connectionStatus === "connected" && modelReady === false && (
        <div className="app-banner app-banner--model-missing" role="status">
          Model checkpoint not loaded — run training or place lstm_best.keras in artifacts/checkpoints/.
        </div>
      )}

      <main className="app-main">
        <SignerPanel
          onPrediction={handlePrediction}
          onConnectionChange={setConnectionStatus}
        />
        <HearingPanel onSpeech={handleSpeech} />
      </main>
      <ConversationLog entries={log} />
    </div>
  );
}
