import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Alert } from "./alert";

describe("Alert", () => {
  it("renders with role='alert'", () => {
    render(<Alert>body</Alert>);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("renders the title and body when provided", () => {
    render(<Alert title="Heads up">Details here</Alert>);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Heads up");
    expect(alert).toHaveTextContent("Details here");
  });

  it("applies tone classes for each variant", () => {
    const tones = ["info", "success", "warning", "danger", "neutral"] as const;
    for (const tone of tones) {
      const { unmount } = render(<Alert tone={tone}>x</Alert>);
      const cls = screen.getByRole("alert").className;
      if (tone === "neutral") {
        expect(cls).toMatch(/bg-\[var\(--color-surface-sunken\)\]/);
      } else {
        expect(cls).toMatch(new RegExp(`bg-\\[var\\(--color-${tone}-subtle\\)\\]`));
      }
      unmount();
    }
  });

  it("defaults to the info tone", () => {
    render(<Alert>x</Alert>);
    expect(screen.getByRole("alert").className).toMatch(
      /bg-\[var\(--color-info-subtle\)\]/,
    );
  });

  it("merges user className with variant classes", () => {
    render(<Alert className="custom">x</Alert>);
    expect(screen.getByRole("alert")).toHaveClass("custom");
  });

  it("omits the title node when no title prop is given", () => {
    render(<Alert>only-body</Alert>);
    // The only paragraph node should be absent — the body is rendered in a div.
    expect(screen.queryByText("only-body")?.tagName).not.toBe("P");
  });
});
