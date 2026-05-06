import { SignerPanel } from "./components/SignerPanel";
import { HearingPanel } from "./components/HearingPanel";
import "./App.css";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-logo" aria-hidden="true">🤟</span>
        <h1>SignLearn</h1>
      </header>
      <main className="app-main">
        <SignerPanel />
        <HearingPanel />
      </main>
    </div>
  );
}
