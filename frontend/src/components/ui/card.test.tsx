import { describe, it, expect } from "vitest";
import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "./card";

describe("Card composition", () => {
  it("renders the full card composition", () => {
    render(
      <Card data-testid="root">
        <CardHeader>
          <CardTitle>Title</CardTitle>
          <CardDescription>Desc</CardDescription>
        </CardHeader>
        <CardContent>Body</CardContent>
        <CardFooter>
          <button>Go</button>
        </CardFooter>
      </Card>,
    );

    expect(screen.getByTestId("root")).toBeInTheDocument();
    const heading = screen.getByRole("heading", { name: "Title", level: 3 });
    expect(heading).toBeInTheDocument();
    expect(screen.getByText("Desc")).toBeInTheDocument();
    expect(screen.getByText("Body")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Go" })).toBeInTheDocument();
  });

  it("CardTitle is an h3", () => {
    render(<CardTitle>t</CardTitle>);
    expect(screen.getByText("t").tagName).toBe("H3");
  });

  it("CardDescription is a p", () => {
    render(<CardDescription>d</CardDescription>);
    expect(screen.getByText("d").tagName).toBe("P");
  });

  it("merges user className with base classes", () => {
    render(<Card className="custom-root" data-testid="c">x</Card>);
    const node = screen.getByTestId("c");
    expect(node).toHaveClass("custom-root");
    expect(node.className).toMatch(/border/);
  });

  it("forwards refs on Card", () => {
    const ref = createRef<HTMLDivElement>();
    render(<Card ref={ref}>x</Card>);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });
});
