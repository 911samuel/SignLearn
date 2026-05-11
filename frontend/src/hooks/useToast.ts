import { createContext, useContext } from "react";

export type ToastKind = "info" | "success" | "error" | "warn";

export interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
}

export interface ToastAPI {
  info: (msg: string) => void;
  success: (msg: string) => void;
  error: (msg: string) => void;
  warn: (msg: string) => void;
}

export const ToastContext = createContext<ToastAPI>({
  info: () => {},
  success: () => {},
  error: () => {},
  warn: () => {},
});

export function useToast(): ToastAPI {
  return useContext(ToastContext);
}
