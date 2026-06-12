"use client";

import { Settings2 } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
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
      <PopoverContent className="w-80">
        <h3 className="text-sm font-semibold text-[var(--color-text)]">{t("a11y.menu.title")}</h3>
        <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
          Settings save to this device.
        </p>

        <div className="mt-4 space-y-5">
          <div>
            <Label className="mb-2">{t("a11y.theme.label")}</Label>
            <ToggleGroup
              type="single"
              value={prefs.theme}
              onValueChange={(v) => v && update({ theme: v as typeof prefs.theme })}
              aria-label={t("a11y.theme.label")}
              className="flex-wrap"
            >
              <ToggleGroupItem value="system">{t("a11y.theme.system")}</ToggleGroupItem>
              <ToggleGroupItem value="light">{t("a11y.theme.light")}</ToggleGroupItem>
              <ToggleGroupItem value="dark">{t("a11y.theme.dark")}</ToggleGroupItem>
              <ToggleGroupItem value="high-contrast">{t("a11y.theme.highContrast")}</ToggleGroupItem>
            </ToggleGroup>
          </div>

          <div>
            <Label className="mb-2">{t("a11y.textSize.label")}</Label>
            <ToggleGroup
              type="single"
              value={prefs.textSize}
              onValueChange={(v) => v && update({ textSize: v as typeof prefs.textSize })}
              aria-label={t("a11y.textSize.label")}
            >
              <ToggleGroupItem value="normal">{t("a11y.textSize.normal")}</ToggleGroupItem>
              <ToggleGroupItem value="large">{t("a11y.textSize.large")}</ToggleGroupItem>
              <ToggleGroupItem value="xlarge">{t("a11y.textSize.xlarge")}</ToggleGroupItem>
            </ToggleGroup>
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="reduce-motion">{t("a11y.motion.label")}</Label>
            <Switch
              id="reduce-motion"
              checked={prefs.reduceMotion}
              onCheckedChange={(v) => update({ reduceMotion: v })}
            />
          </div>

          <div>
            <Label className="mb-2">{t("a11y.ptt.label")}</Label>
            <ToggleGroup
              type="single"
              value={prefs.pushToTalkMode}
              onValueChange={(v) => v && update({ pushToTalkMode: v as typeof prefs.pushToTalkMode })}
              aria-label={t("a11y.ptt.label")}
            >
              <ToggleGroupItem value="hold">{t("a11y.ptt.hold")}</ToggleGroupItem>
              <ToggleGroupItem value="toggle">{t("a11y.ptt.toggle")}</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
