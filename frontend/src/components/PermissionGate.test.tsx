import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { PermissionGate } from "./PermissionGate";

describe("PermissionGate copy by kind", () => {
  it("renders camera-specific copy when kind='camera'", () => {
    render(<PermissionGate kind="camera" onAllow={vi.fn()} />);
    expect(
      screen.getByRole("heading", { name: /turn on your camera/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /allow camera & start/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/21 hand landmarks per hand/i)).toBeInTheDocument();
  });

  it("renders microphone-specific copy when kind='microphone'", () => {
    render(<PermissionGate kind="microphone" onAllow={vi.fn()} />);
    expect(
      screen.getByRole("heading", { name: /turn on your microphone/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /allow microphone & start/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/web speech api/i)).toBeInTheDocument();
  });
});

describe("PermissionGate allow flow", () => {
  it("calls onAllow when the primary button is clicked", async () => {
    const onAllow = vi.fn().mockResolvedValue(undefined);
    render(<PermissionGate kind="camera" onAllow={onAllow} />);
    fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
    await waitFor(() => expect(onAllow).toHaveBeenCalledOnce());
  });

  it("shows 'Requesting…' and disables the button while the request is pending", async () => {
    let resolve!: () => void;
    const onAllow = vi.fn(
      () => new Promise<void>((r) => {
        resolve = r;
      }),
    );
    render(<PermissionGate kind="camera" onAllow={onAllow} />);

    const btn = screen.getByRole("button", { name: /allow camera & start/i });
    fireEvent.click(btn);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /requesting/i })).toBeDisabled();
    });

    resolve();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /allow camera & start/i })).not.toBeDisabled(),
    );
  });

  it("surfaces the error message when onAllow rejects with an Error", async () => {
    const onAllow = vi.fn().mockRejectedValue(new Error("NotAllowedError: blocked"));
    render(<PermissionGate kind="camera" onAllow={onAllow} />);
    fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
    await waitFor(() =>
      expect(screen.getByText(/NotAllowedError: blocked/)).toBeInTheDocument(),
    );
  });

  it("falls back to a generic message when the rejection isn't an Error", async () => {
    const onAllow = vi.fn().mockRejectedValue("nope");
    render(<PermissionGate kind="camera" onAllow={onAllow} />);
    fireEvent.click(screen.getByRole("button", { name: /allow camera & start/i }));
    await waitFor(() =>
      expect(
        screen.getByText(/Permission was denied or unavailable/),
      ).toBeInTheDocument(),
    );
  });

  it("clears a previously shown error on retry", async () => {
    const onAllow = vi
      .fn()
      .mockRejectedValueOnce(new Error("first"))
      .mockResolvedValueOnce(undefined);
    render(<PermissionGate kind="camera" onAllow={onAllow} />);

    const btn = screen.getByRole("button", { name: /allow camera & start/i });
    fireEvent.click(btn);
    await waitFor(() => expect(screen.getByText(/first/)).toBeInTheDocument());

    fireEvent.click(btn);
    await waitFor(() => expect(screen.queryByText(/first/)).toBeNull());
  });
});

describe("PermissionGate secondary action", () => {
  it("is hidden when secondaryAction is not provided", () => {
    render(<PermissionGate kind="camera" onAllow={vi.fn()} />);
    // Only the allow button should be present.
    expect(screen.getAllByRole("button")).toHaveLength(1);
  });

  it("calls secondaryAction.onClick when its button is clicked", () => {
    const onClick = vi.fn();
    render(
      <PermissionGate
        kind="camera"
        onAllow={vi.fn()}
        secondaryAction={{ label: "Skip for now", onClick }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /skip for now/i }));
    expect(onClick).toHaveBeenCalledOnce();
  });
});
