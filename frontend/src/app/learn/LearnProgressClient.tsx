"use client";

import { Flame, Sparkles, Star, Trophy } from "lucide-react";
import { useProgress } from "@/lib/progress";
import { StatCard } from "@/components/primitives/StatCard";
import { ALL_LESSONS } from "@/data/curriculum";

export function LearnProgressClient() {
  const progress = useProgress();
  const completedCount = Object.values(progress.lessons).filter((l) => l.completed).length;
  const totalLessons = ALL_LESSONS.length;
  const accuracy =
    progress.attempts.length === 0
      ? 0
      : progress.attempts.filter((a) => a.correct).length / progress.attempts.length;

  return (
    <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        label="Lessons completed"
        value={`${completedCount} / ${totalLessons}`}
        hint="Across all units"
        icon={<Trophy className="size-5" />}
      />
      <StatCard
        label="XP earned"
        value={progress.xp.toLocaleString()}
        hint="10 XP per correct sign · 25 XP per lesson"
        icon={<Sparkles className="size-5" />}
      />
      <StatCard
        label="Day streak"
        value={progress.streakDays}
        hint={progress.streakDays === 0 ? "Sign today to start a streak" : "Keep it going!"}
        icon={<Flame className="size-5" />}
      />
      <StatCard
        label="Recognition accuracy"
        value={progress.attempts.length === 0 ? "—" : `${Math.round(accuracy * 100)}%`}
        hint={`${progress.attempts.length} attempts logged`}
        icon={<Star className="size-5" />}
      />
    </div>
  );
}
