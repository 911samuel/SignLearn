import { cn } from "@/lib/utils";

export interface SectionHeaderProps {
  eyebrow?: string;
  title: React.ReactNode;
  description?: React.ReactNode;
  align?: "left" | "center";
  className?: string;
  as?: "h1" | "h2" | "h3";
}

export function SectionHeader({
  eyebrow,
  title,
  description,
  align = "left",
  className,
  as: H = "h2",
}: SectionHeaderProps) {
  return (
    <div
      className={cn(
        "max-w-3xl",
        align === "center" ? "mx-auto text-center" : "text-left",
        className,
      )}
    >
      {eyebrow && <p className="eyebrow mb-3">{eyebrow}</p>}
      <H className={cn(H === "h1" ? "heading-display" : H === "h2" ? "heading-h1" : "heading-h2", "text-[var(--color-text)] mb-4")}>{title}</H>
      {description && (
        <p className="text-lg text-[var(--color-text-muted)] leading-relaxed">{description}</p>
      )}
    </div>
  );
}
