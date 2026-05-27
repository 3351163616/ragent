import { create } from "zustand";

import { storage } from "@/utils/storage";

export type ChatEdition = "legacy" | "anthropic";

interface EditionState {
  edition: ChatEdition;
  setEdition: (edition: ChatEdition) => void;
  toggleEdition: () => void;
  initialize: () => void;
}

const DEFAULT_EDITION: ChatEdition = "legacy";

function isValidEdition(value: string | null): value is ChatEdition {
  return value === "legacy" || value === "anthropic";
}

export const useEditionStore = create<EditionState>((set, get) => ({
  edition: DEFAULT_EDITION,
  setEdition: (edition) => {
    storage.setEdition(edition);
    set({ edition });
  },
  toggleEdition: () => {
    const next: ChatEdition = get().edition === "legacy" ? "anthropic" : "legacy";
    get().setEdition(next);
  },
  initialize: () => {
    const stored = storage.getEdition();
    const edition: ChatEdition = isValidEdition(stored) ? stored : DEFAULT_EDITION;
    set({ edition });
  }
}));
