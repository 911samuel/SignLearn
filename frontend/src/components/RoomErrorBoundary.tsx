"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  onLeave?: () => void;
}

interface State {
  error: Error | null;
}

export class RoomErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[RoomErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={styles.wrapper} role="alert">
          <p style={styles.heading}>Something went wrong in the room.</p>
          <p style={styles.message}>{this.state.error.message}</p>
          <div style={styles.actions}>
            <button style={styles.btn} onClick={() => this.setState({ error: null })}>
              Try again
            </button>
            {this.props.onLeave && (
              <button style={{ ...styles.btn, ...styles.btnSecondary }} onClick={this.props.onLeave}>
                Leave room
              </button>
            )}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    flex: 1, gap: "0.75rem", padding: "2rem", textAlign: "center",
  },
  heading: { fontSize: "1.1rem", fontWeight: 600, color: "var(--danger)", margin: 0 },
  message: { fontSize: "0.9rem", color: "var(--text-muted)", margin: 0, maxWidth: 480 },
  actions: { display: "flex", gap: "0.75rem", marginTop: "0.5rem" },
  btn: {
    padding: "0.5rem 1.25rem", borderRadius: 6, border: "none",
    background: "var(--primary)", color: "#fff", cursor: "pointer",
    fontFamily: "inherit", fontSize: "0.9rem",
  },
  btnSecondary: { background: "transparent", border: "1px solid var(--border)", color: "var(--text-muted)" },
};
