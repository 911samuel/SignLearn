import en from "./en.json";

type Dict = Record<string, string>;
const dictionaries: Record<string, Dict> = { en };

let currentLocale = "en";

export function setLocale(locale: string) {
  if (dictionaries[locale]) currentLocale = locale;
}

export function t(key: string, fallback?: string): string {
  const dict = dictionaries[currentLocale] || en;
  return dict[key] ?? fallback ?? key;
}
