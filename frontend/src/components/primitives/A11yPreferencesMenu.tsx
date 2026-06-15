"use client";

import { HardDrive, Hand, Palette, Settings2, Sparkles, Type } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { usePreferences } from "@/lib/preferences";
import { t } from "@/i18n";

export function A11yPreferencesMenu() {
  const { prefs, update } = usePreferences();

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" aria-label={t("a11y.menu.title")}>
          <Settings2 className="size-5" aria-hidden />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[22rem] p-0" align="end">
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-[var(--color-border)] px-4 py-3">
          <span
            aria-hidden
            className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--color-brand-subtle)] text-[var(--color-brand-subtle-foreground)]"
          >
            <Settings2 className="size-4" />
          </span>
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-[var(--color-text)]">
              {t("a11y.menu.title")}
            </h3>
            <p className="text-xs text-[var(--color-text-muted)]">
              Tune the app to your needs
            </p>
          </div>
        </div>

        {/* Sections */}
        <div className="divide-y divide-[var(--color-border)]">
          <Section icon={<Palette className="size-4" aria-hidden />} label={t("a11y.theme.label")}>
            <ToggleGroup
              type="single"
              value={prefs.theme}
              onValueChange={(v) => v && update({ theme: v as typeof prefs.theme })}
              aria-label={t("a11y.theme.label")}
              className="flex-wrap gap-1.5"
            >
              <ToggleGroupItem value="system">{t("a11y.theme.system")}</ToggleGroupItem>
              <ToggleGroupItem value="light">{t("a11y.theme.light")}</ToggleGroupItem>
              <ToggleGroupItem value="dark">{t("a11y.theme.dark")}</ToggleGroupItem>
              <ToggleGroupItem value="high-contrast">
                {t("a11y.theme.highContrast")}
              </ToggleGroupItem>
            </ToggleGroup>
          </Section>

          <Section icon={<Type className="size-4" aria-hidden />} label={t("a11y.textSize.label")}>
            <ToggleGroup
              type="single"
              value={prefs.textSize}
              onValueChange={(v) => v && update({ textSize: v as typeof prefs.textSize })}
              aria-label={t("a11y.textSize.label")}
              className="gap-1.5"
            >
              <ToggleGroupItem value="normal">{t("a11y.textSize.normal")}</ToggleGroupItem>
              <ToggleGroupItem value="large">{t("a11y.textSize.large")}</ToggleGroupItem>
              <ToggleGroupItem value="xlarge">{t("a11y.textSize.xlarge")}</ToggleGroupItem>
            </ToggleGroup>
          </Section>

          <Section
            icon={<Sparkles className="size-4" aria-hidden />}
            label={t("a11y.motion.label")}
            inline
          >
            <Switch
              id="reduce-motion"
              checked={prefs.reduceMotion}
              onCheckedChange={(v) => update({ reduceMotion: v })}
              aria-label={t("a11y.motion.label")}
            />
          </Section>

          <Section icon={<Hand className="size-4" aria-hidden />} label={t("a11y.ptt.label")}>
            <ToggleGroup
              type="single"
              value={prefs.pushToTalkMode}
              onValueChange={(v) => v && update({ pushToTalkMode: v as typeof prefs.pushToTalkMode })}
              aria-label={t("a11y.ptt.label")}
              className="gap-1.5"
            >
              <ToggleGroupItem value="hold">{t("a11y.ptt.hold")}</ToggleGroupItem>
              <ToggleGroupItem value="toggle">{t("a11y.ptt.toggle")}</ToggleGroupItem>
            </ToggleGroup>
          </Section>
        </div>

        {/* Footer disclaimer */}
        <div className="flex items-center gap-2 border-t border-[var(--color-border)] bg-[var(--color-surface-sunken)] px-4 py-2.5">
          <HardDrive className="size-3.5 shrink-0 text-[var(--color-text-faint)]" aria-hidden />
          <p className="text-xs text-[var(--color-text-muted)]">
            Saved on this device — synced nowhere else.
          </p>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function Section({
  icon,
  label,
  children,
  inline = false,
}: {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
  inline?: boolean;
}) {
  return (
    <div className="px-4 py-3.5">
      <div
        className={
          inline
            ? "flex items-center justify-between gap-3"
            : "flex flex-col gap-2.5"
        }
      >
        <div className="inline-flex items-center gap-2 text-[var(--color-text-muted)]">
          {icon}
          <span className="text-xs font-semibold uppercase tracking-wider">{label}</span>
        </div>
        <div className={inline ? "" : "w-full"}>{children}</div>
      </div>
    </div>
  );
}
