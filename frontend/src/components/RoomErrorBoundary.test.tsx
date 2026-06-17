import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RoomErrorBoundary } from "./RoomErrorBoundary";

function Boom({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("kaboom");
  return <div>safe content</div>;
}

let errorSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  errorSpy.mockRestore();
});

describe("RoomErrorBoundary", () => {
  it("renders children when no error is thrown", () => {
    render(
      <RoomErrorBoundary>
        <Boom shouldThrow={false} />
      </RoomErrorBoundary>,
    );
    expect(screen.getByText("safe content")).toBeInTheDocument();
  });

  it("renders the fallback with the error message on throw", () => {
    render(
      <RoomErrorBoundary>
        <Boom shouldThrow={true} />
      </RoomErrorBoundary>,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Something went wrong in the room.");
    expect(alert).toHaveTextContent("kaboom");
  });

  it("hides the 'Leave room' button unless onLeave is provided", () => {
    render(
      <RoomErrorBoundary>
        <Boom shouldThrow={true} />
      </RoomErrorBoundary>,
    );
    expect(screen.queryByRole("button", { name: /leave room/i })).toBeNull();
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  it("calls onLeave when the leave button is clicked", () => {
    const onLeave = vi.fn();
    render(
      <RoomErrorBoundary onLeave={onLeave}>
        <Boom shouldThrow={true} />
      </RoomErrorBoundary>,
    );
    fireEvent.click(screen.getByRole("button", { name: /leave room/i }));
    expect(onLeave).toHaveBeenCalledOnce();
  });

  it("clears the error state when 'Try again' is clicked", () => {
    const { rerender } = render(
      <RoomErrorBoundary>
        <Boom shouldThrow={true} />
      </RoomErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    rerender(
      <RoomErrorBoundary>
        <Boom shouldThrow={false} />
      </RoomErrorBoundary>,
    );
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(screen.getByText("safe content")).toBeInTheDocument();
  });

  it("logs the error to console.error", () => {
    render(
      <RoomErrorBoundary>
        <Boom shouldThrow={true} />
      </RoomErrorBoundary>,
    );
    expect(errorSpy).toHaveBeenCalled();
  });
});
