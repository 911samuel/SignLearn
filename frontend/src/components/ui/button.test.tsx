import { describe, it, expect, vi } from "vitest";
import { createRef } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { Button } from "./button";

describe("Button", () => {
  it("renders a native <button> by default", () => {
    render(<Button>Hi</Button>);
    const btn = screen.getByRole("button", { name: "Hi" });
    expect(btn.tagName).toBe("BUTTON");
  });

  it("applies the primary variant class by default", () => {
    render(<Button>p</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-[var(--color-brand)]");
  });

  it("applies a chosen variant class", () => {
    render(<Button variant="danger">x</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-red-700");
  });

  it("applies size classes", () => {
    render(<Button size="sm">s</Button>);
    expect(screen.getByRole("button").className).toMatch(/\bh-9\b/);
  });

  it("merges a user-supplied className over defaults", () => {
    render(<Button className="extra-class">x</Button>);
    expect(screen.getByRole("button")).toHaveClass("extra-class");
  });

  it("forwards the ref to the underlying button", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>r</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("forwards arbitrary props (type, onClick, disabled)", () => {
    const onClick = vi.fn();
    render(
      <Button type="submit" onClick={onClick} disabled>
        s
      </Button>,
    );
    const btn = screen.getByRole("button") as HTMLButtonElement;
    expect(btn.type).toBe("submit");
    expect(btn.disabled).toBe(true);
    fireEvent.click(btn);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("renders the child element via Slot when asChild=true", () => {
    render(
      <Button asChild>
        <a href="/x">link</a>
      </Button>,
    );
    const link = screen.getByRole("link", { name: "link" });
    expect(link.tagName).toBe("A");
    expect(link).toHaveClass("bg-[var(--color-brand)]");
  });
});
