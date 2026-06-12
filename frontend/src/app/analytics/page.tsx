"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, Clock, Cpu, Database, Hand, RefreshCw, Server, TrendingUp } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
} from "recharts";
import { PageShell } from "@/components/primitives/PageShell";
import { SectionHeader } from "@/components/primitives/SectionHeader";
import { StatCard } from "@/components/primitives/StatCard";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type HealthResponse, type MetricsResponse } from "@/lib/api";
import { useProgress } from "@/lib/progress";

export default function AnalyticsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const progress = useProgress();

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [h, m] = await Promise.all([api.health(), api.metrics()]);
      setHealth(h);
      setMetrics(m);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load metrics");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15_000);
    return () => clearInterval(id);
  }, []);

  // Per-sign accuracy from local attempts.
  const accuracyBySign = useMemo(() => {
    const map = new Map<string, { hits: number; misses: number }>();
    for (const a of progress.attempts) {
      const e = map.get(a.sign) ?? { hits: 0, misses: 0 };
      if (a.correct) e.hits += 1;
      else e.misses += 1;
      map.set(a.sign, e);
    }
    return Array.from(map.entries())
      .map(([sign, v]) => ({
        sign,
        accuracy: Math.round((v.hits / (v.hits + v.misses)) * 100),
        attempts: v.hits + v.misses,
      }))
      .filter((x) => x.attempts >= 3)
      .sort((a, b) => b.attempts - a.attempts)
      .slice(0, 12);
  }, [progress.attempts]);

  // Last-30 attempts as a rolling timeseries.
  const timeline = useMemo(() => {
    const last = progress.attempts.slice(-30);
    return last.map((a, i) => ({
      i: i + 1,
      ok: a.correct ? 1 : 0,
      conf: Math.round(a.confidence * 100),
    }));
  }, [progress.attempts]);

  return (
    <PageShell>
      <div className="flex flex-wrap items-end justify-between gap-4 pt-10">
        <SectionHeader eyebrow="Analytics" title="How the recogniser and your practice are doing." description="Live metrics from this SignLearn server and from your practice on this device." as="h1" />
        <Button variant="secondary" onClick={refresh}>
          <RefreshCw className="size-4" aria-hidden />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert tone="warning" title="Couldn't reach the SignLearn server" className="mt-6">
          {error}. Showing local-only stats below — start the backend with <code className="rounded bg-[var(--color-surface-sunken)] px-1.5 py-0.5 font-mono text-[0.85em]">make serve</code> to see live system metrics.
        </Alert>
      )}

      {/* SYSTEM */}
      <section aria-labelledby="system-metrics" className="mt-10">
        <h2 id="system-metrics" className="heading-h2 text-[var(--color-text)]">
          System
        </h2>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          Live from this server&apos;s <code className="font-mono">/health</code> and <code className="font-mono">/metrics</code> endpoints.
        </p>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {loading && !health ? (
            <>
              <Skeleton className="h-32" />
              <Skeleton className="h-32" />
              <Skeleton className="h-32" />
              <Skeleton className="h-32" />
            </>
          ) : (
            <>
              <StatCard
                label="Server status"
                value={
                  <span className="inline-flex items-center gap-2">
                    {health?.status === "ok" ? (
                      <Badge tone="success">Healthy</Badge>
                    ) : (
                      <Badge tone="warning">{health?.status ?? "unknown"}</Badge>
                    )}
                  </span>
                }
                hint={
                  health?.uptime_seconds
                    ? `Up ${(health.uptime_seconds / 3600).toFixed(1)} h`
                    : "Uptime unavailable"
                }
                icon={<Server className="size-5" />}
              />
              <StatCard
                label="Inference backend"
                value={health?.backend ?? "—"}
                hint={`${health?.num_classes ?? "?"} classes`}
                icon={<Cpu className="size-5" />}
              />
              <StatCard
                label="P95 latency"
                value={
                  metrics?.prediction_latency_ms_p95 != null
                    ? `${metrics.prediction_latency_ms_p95.toFixed(1)} ms`
                    : "—"
                }
                hint="Per inference"
                icon={<Clock className="size-5" />}
              />
              <StatCard
                label="Predictions / min"
                value={
                  metrics?.predictions_per_minute != null
                    ? Math.round(metrics.predictions_per_minute)
                    : metrics?.predictions_total ?? "—"
                }
                hint="Across all rooms"
                icon={<Activity className="size-5" />}
              />
            </>
          )}
        </div>

        {health?.model_sha && (
          <Card className="mt-4 flex flex-wrap items-center gap-3 p-4 text-sm">
            <Database className="size-4 text-[var(--color-text-muted)]" aria-hidden />
            <span className="text-[var(--color-text-muted)]">Model SHA</span>
            <code className="rounded bg-[var(--color-surface-sunken)] px-2 py-0.5 font-mono text-[0.85em] text-[var(--color-text)]">
              {health.model_sha.slice(0, 16)}…
            </code>
            <span className="ml-auto text-xs text-[var(--color-text-muted)]">
              Updated every 15 s · auto-refresh on
            </span>
          </Card>
        )}
      </section>

      {/* YOU */}
      <section aria-labelledby="you-metrics" className="mt-12 pb-16">
        <h2 id="you-metrics" className="heading-h2 text-[var(--color-text)]">
          You (this device)
        </h2>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          From your practice history, stored locally in your browser.
        </p>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Total XP" value={progress.xp.toLocaleString()} icon={<TrendingUp className="size-5" />} />
          <StatCard label="Day streak" value={progress.streakDays} icon={<Hand className="size-5" />} />
          <StatCard
            label="Total attempts"
            value={progress.attempts.length}
            icon={<Activity className="size-5" />}
          />
          <StatCard
            label="Accuracy"
            value={
              progress.attempts.length === 0
                ? "—"
                : `${Math.round(
                    (progress.attempts.filter((a) => a.correct).length / progress.attempts.length) *
                      100,
                  )}%`
            }
            icon={<AlertTriangle className="size-5" />}
          />
        </div>

        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <Card className="p-5">
            <p className="eyebrow">Confidence on your last attempts</p>
            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
              Each point is one sign attempt; height shows the recogniser&apos;s confidence.
            </p>
            <div className="mt-4 h-56">
              {timeline.length === 0 ? (
                <p className="grid h-full place-items-center text-sm text-[var(--color-text-faint)]">
                  No attempts yet — head to <a href="/practice">/practice</a> to log some.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="i" stroke="var(--color-text-muted)" fontSize={12} />
                    <YAxis stroke="var(--color-text-muted)" fontSize={12} domain={[0, 100]} unit="%" />
                    <RTooltip
                      contentStyle={{
                        background: "var(--color-surface-elevated)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 10,
                        color: "var(--color-text)",
                      }}
                      formatter={(v) => `${v}%`}
                    />
                    <Line type="monotone" dataKey="conf" stroke="var(--color-brand)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </Card>

          <Card className="p-5">
            <p className="eyebrow">Accuracy by sign (≥ 3 attempts)</p>
            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
              Where the recogniser most often agrees with what you signed.
            </p>
            <div className="mt-4 h-56">
              {accuracyBySign.length === 0 ? (
                <p className="grid h-full place-items-center text-sm text-[var(--color-text-faint)]">
                  Practice each sign at least 3 times to see it here.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={accuracyBySign}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="sign" stroke="var(--color-text-muted)" fontSize={12} />
                    <YAxis stroke="var(--color-text-muted)" fontSize={12} domain={[0, 100]} unit="%" />
                    <RTooltip
                      contentStyle={{
                        background: "var(--color-surface-elevated)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 10,
                        color: "var(--color-text)",
                      }}
                      formatter={(v) => `${v}%`}
                    />
                    <Bar dataKey="accuracy" fill="var(--color-brand)" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </Card>
        </div>
      </section>
    </PageShell>
  );
}
