import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Badge } from "./badge";

describe("Badge", () => {
  it("renders a span with its children", () => {
    render(<Badge>new</Badge>);
    const el = screen.getByText("new");
    expect(el.tagName).toBe("SPAN");
  });

  it("defaults to the neutral tone", () => {
    render(<Badge>n</Badge>);
    expect(screen.getByText("n").className).toMatch(
      /bg-\[var\(--color-surface-sunken\)\]/,
    );
  });

  it("applies the requested tone", () => {
    render(<Badge tone="brand">b</Badge>);
    expect(screen.getByText("b").className).toMatch(
      /bg-\[var\(--color-brand-subtle\)\]/,
    );
  });

  it("appends a user-supplied className", () => {
    render(<Badge className="my-badge">x</Badge>);
    expect(screen.getByText("x")).toHaveClass("my-badge");
  });

  it("forwards span attributes", () => {
    render(<Badge data-testid="b" aria-label="label">x</Badge>);
    const el = screen.getByTestId("b");
    expect(el).toHaveAttribute("aria-label", "label");
  });
});
