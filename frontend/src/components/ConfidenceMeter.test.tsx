import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConfidenceMeter } from "./ConfidenceMeter";

describe("ConfidenceMeter status text", () => {
  it("says 'Signing paused' when paused", () => {
    render(<ConfidenceMeter value={0.9} label="hello" ready={true} paused={true} />);
    expect(screen.getByRole("status")).toHaveTextContent("Signing paused");
  });

  it("says 'Buffering frames…' when not ready", () => {
    render(<ConfidenceMeter value={null} label={null} ready={false} paused={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("Buffering frames");
  });

  it("says 'Hold steady — I'm listening' when ready but no label", () => {
    render(<ConfidenceMeter value={0.2} label={null} ready={true} paused={false} />);
    expect(screen.getByRole("status")).toHaveTextContent(/Hold steady — I'm listening/);
  });

  it("shows tentative '<label>? Hold steady…' below threshold", () => {
    render(<ConfidenceMeter value={0.5} label="cat" ready={true} paused={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("cat? Hold steady");
  });

  it("commits to bare label when above threshold", () => {
    render(<ConfidenceMeter value={0.95} label="cat" ready={true} paused={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("cat");
    expect(screen.getByRole("status")).not.toHaveTextContent("Hold steady");
  });

  it("respects a custom threshold", () => {
    render(
      <ConfidenceMeter value={0.5} label="x" ready={true} paused={false} threshold={0.4} />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(/^x/);
  });
});

describe("ConfidenceMeter percentage", () => {
  it("rounds and shows pct when ready and label present", () => {
    render(<ConfidenceMeter value={0.876} label="ok" ready={true} paused={false} />);
    expect(screen.getByText("88%")).toBeInTheDocument();
  });

  it("clamps negative values to 0%", () => {
    render(<ConfidenceMeter value={-0.5} label="ok" ready={true} paused={false} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("clamps values above 1 to 100%", () => {
    render(<ConfidenceMeter value={1.7} label="ok" ready={true} paused={false} />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("treats a null value as 0", () => {
    render(<ConfidenceMeter value={null} label="ok" ready={true} paused={false} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("hides percentage when not ready or no label", () => {
    render(<ConfidenceMeter value={0.9} label={null} ready={true} paused={false} />);
    expect(screen.queryByText(/%/)).toBeNull();
  });

  it("exposes pct in the aria-label", () => {
    render(<ConfidenceMeter value={0.5} label="ok" ready={true} paused={false} />);
    expect(screen.getByRole("status")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("50 percent") as unknown as string,
    );
  });
});

describe("ConfidenceMeter correction flow", () => {
  it("hides the 'Not what I signed' button unless committed", () => {
    const onCorrect = vi.fn();
    render(
      <ConfidenceMeter
        value={0.3}
        label="cat"
        ready={true}
        paused={false}
        onCorrect={onCorrect}
      />,
    );
    expect(screen.queryByRole("button", { name: /not what you signed/i })).toBeNull();
  });

  it("hides the button when committed but no onCorrect handler", () => {
    render(<ConfidenceMeter value={0.95} label="cat" ready={true} paused={false} />);
    expect(screen.queryByRole("button", { name: /not what you signed/i })).toBeNull();
  });

  it("opens the correction form, submits, and calls onCorrect", () => {
    const onCorrect = vi.fn();
    render(
      <ConfidenceMeter
        value={0.95}
        label="cat"
        ready={true}
        paused={false}
        onCorrect={onCorrect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /not what you signed/i }));
    const input = screen.getByLabelText(/what did you sign/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "dog" } });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));
    expect(onCorrect).toHaveBeenCalledWith("cat", "dog");
  });

  it("disables Send when the input is empty or whitespace", () => {
    render(
      <ConfidenceMeter
        value={0.95}
        label="cat"
        ready={true}
        paused={false}
        onCorrect={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /not what you signed/i }));
    const send = screen.getByRole("button", { name: /^send$/i }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText(/what did you sign/i), {
      target: { value: "   " },
    });
    expect(send.disabled).toBe(true);
  });

  it("does not call onCorrect for whitespace-only submissions", () => {
    const onCorrect = vi.fn();
    render(
      <ConfidenceMeter
        value={0.95}
        label="cat"
        ready={true}
        paused={false}
        onCorrect={onCorrect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /not what you signed/i }));
    const input = screen.getByLabelText(/what did you sign/i);
    fireEvent.change(input, { target: { value: "ok" } });
    fireEvent.change(input, { target: { value: "   " } });
    const form = input.closest("form")!;
    fireEvent.submit(form);
    expect(onCorrect).not.toHaveBeenCalled();
  });

  it("Cancel closes the form without firing onCorrect", () => {
    const onCorrect = vi.fn();
    render(
      <ConfidenceMeter
        value={0.95}
        label="cat"
        ready={true}
        paused={false}
        onCorrect={onCorrect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /not what you signed/i }));
    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));
    expect(screen.queryByLabelText(/what did you sign/i)).toBeNull();
    expect(onCorrect).not.toHaveBeenCalled();
  });

  it("closes the correction form when the label changes", () => {
    const onCorrect = vi.fn();
    const { rerender } = render(
      <ConfidenceMeter
        value={0.95}
        label="cat"
        ready={true}
        paused={false}
        onCorrect={onCorrect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /not what you signed/i }));
    expect(screen.getByLabelText(/what did you sign/i)).toBeInTheDocument();

    rerender(
      <ConfidenceMeter
        value={0.95}
        label="dog"
        ready={true}
        paused={false}
        onCorrect={onCorrect}
      />,
    );
    expect(screen.queryByLabelText(/what did you sign/i)).toBeNull();
  });
});
