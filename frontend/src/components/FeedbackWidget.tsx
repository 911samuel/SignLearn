"use client";

import { useRef, useState } from "react";
import { Accessibility, Bug, Lightbulb, MessageCirclePlus, Sparkles, X } from "lucide-react";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const CATEGORIES = [
  { value: "bug",            label: "Something broke",   icon: Bug },
  { value: "praise",         label: "This helped me",    icon: Sparkles },
  { value: "idea",           label: "Feature idea",      icon: Lightbulb },
  { value: "accessibility",  label: "Accessibility",     icon: Accessibility },
] as const;

type Category = (typeof CATEGORIES)[number]["value"];

export function FeedbackWidget({ roomId }: { roomId?: string }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [category, setCategory] = useState<Category>("bug");
  const [busy, setBusy] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    try {
      await api.feedback({ category, message: text.trim(), context: roomId });
      toast({
        tone: "success",
        title: "Thanks for the feedback",
        description: "We read every message within 48 hours.",
      });
      setText("");
      setOpen(false);
    } catch {
      toast({
        tone: "danger",
        title: "Couldn't send",
        description: "Check your network and try again.",
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        setOpen(v);
        if (v) setTimeout(() => textareaRef.current?.focus(), 80);
      }}
    >
      <DialogTrigger asChild>
        <button
          type="button"
          aria-label="Send feedback"
          className="fixed bottom-20 right-4 z-30 inline-flex h-12 items-center gap-2 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-4 text-sm font-semibold text-[var(--color-text)] shadow-[var(--shadow-overlay)] transition hover:bg-[var(--color-surface-sunken)] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)] md:bottom-4"
        >
          <MessageCirclePlus className="size-4" aria-hidden />
          Feedback
        </button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Send feedback</DialogTitle>
          <DialogDescription>
            Help us make SignLearn work better. We answer every message within 48 hours.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={submit} className="mt-5 space-y-4">
          <fieldset>
            <legend className="text-sm font-semibold text-[var(--color-text)] mb-2">Category</legend>
            <div className="flex flex-wrap gap-2">
              {CATEGORIES.map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setCategory(value)}
                  aria-pressed={category === value}
                  className={cn(
                    "inline-flex h-10 items-center gap-2 rounded-full border px-3 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--color-focus)]",
                    category === value
                      ? "border-transparent bg-[var(--color-brand)] text-[var(--color-brand-foreground)]"
                      : "border-[var(--color-border-strong)] bg-[var(--color-surface)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]",
                  )}
                >
                  <Icon className="size-4" aria-hidden />
                  {label}
                </button>
              ))}
            </div>
          </fieldset>

          <div className="space-y-1.5">
            <Label htmlFor="sl-feedback-text">What's on your mind?</Label>
            <Textarea
              id="sl-feedback-text"
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Tell us what happened, what you expected, or what would help…"
              rows={5}
              required
            />
          </div>

          <DialogFooter>
            <DialogClose asChild>
              <Button variant="ghost" type="button">Cancel</Button>
            </DialogClose>
            <Button type="submit" disabled={busy || !text.trim()}>
              {busy ? "Sending…" : "Send feedback"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
