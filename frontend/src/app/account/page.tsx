"use client";

import { useState } from "react";
import { Download, Trash2, User } from "lucide-react";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Alert } from "@/components/ui/alert";
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
import { usePreferences } from "@/lib/preferences";
import { clearProgress, loadProgress } from "@/lib/progress";
import { useToast } from "@/components/ui/toast";

export default function AccountPage() {
  const { prefs, update, hydrated } = usePreferences();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const { toast } = useToast();

  function exportData() {
    const payload = {
      preferences: prefs,
      progress: loadProgress(),
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `signlearn-data-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast({ tone: "success", title: "Data exported", description: "Saved as JSON to your downloads." });
  }

  function eraseAll() {
    clearProgress();
    try {
      window.localStorage.removeItem("signlearn.prefs.v1");
      window.localStorage.removeItem("signlearn.onboarded.v1");
      window.localStorage.removeItem("signlearn.research.v1");
    } catch {}
    setConfirmOpen(false);
    toast({ tone: "info", title: "All local data cleared" });
    setTimeout(() => window.location.reload(), 600);
  }

  return (
    <PageShell>
      <div className="pt-10 pb-16">
        <SectionHeader
          eyebrow="Your account"
          title="Everything below is stored on this device only."
          description="SignLearn doesn't require an account. Your preferences, progress, and feedback are kept in your browser's local storage."
          as="h1"
        />

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Card className="p-6">
            <div className="mb-5 inline-flex h-12 w-12 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]">
              <User className="size-6" aria-hidden />
            </div>
            <h2 className="heading-h2 text-[var(--color-text)]">Profile</h2>
            <p className="mt-2 text-sm text-[var(--color-text-muted)]">
              Used in room chats. Optional — leave empty to stay anonymous.
            </p>
            <div className="mt-5 space-y-1.5">
              <Label htmlFor="display-name">Display name</Label>
              <Input
                id="display-name"
                value={prefs.name}
                onChange={(e) => update({ name: e.target.value })}
                placeholder="e.g. Alex"
                maxLength={40}
                disabled={!hydrated}
              />
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="heading-h2 text-[var(--color-text)]">Your data on this device</h2>
            <p className="mt-2 text-sm text-[var(--color-text-muted)]">
              Lesson progress, XP, attempts, and accessibility preferences. Nothing leaves this
              browser unless you export it.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <Button variant="secondary" onClick={exportData}>
                <Download className="size-4" aria-hidden /> Export as JSON
              </Button>
              <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
                <DialogTrigger asChild>
                  <Button variant="danger">
                    <Trash2 className="size-4" aria-hidden /> Clear all data
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Clear all SignLearn data?</DialogTitle>
                    <DialogDescription>
                      This erases your progress, preferences, and onboarding state from this
                      browser. It cannot be undone.
                    </DialogDescription>
                  </DialogHeader>
                  <DialogFooter>
                    <DialogClose asChild>
                      <Button variant="ghost">Cancel</Button>
                    </DialogClose>
                    <Button variant="danger" onClick={eraseAll}>
                      Yes, clear it
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
            <Alert tone="info" className="mt-5">
              SignLearn doesn&apos;t have user accounts yet. Backend sync, login, and cross-device
              syncing are on the roadmap.
            </Alert>
          </Card>
        </div>
      </div>
    </PageShell>
  );
}
