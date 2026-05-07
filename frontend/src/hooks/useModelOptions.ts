import * as React from "react";

import { getSystemSettings, type ModelCandidate } from "@/services/settingsService";

export interface ModelOption {
  id: string;
  label: string;
  provider: string;
}

const AUTO_OPTION: ModelOption = { id: "auto", label: "自动选择", provider: "" };

function toOption(candidate: ModelCandidate): ModelOption {
  return {
    id: candidate.id,
    label: candidate.model || candidate.id,
    provider: candidate.provider
  };
}

export function useModelOptions() {
  const [options, setOptions] = React.useState<ModelOption[]>([]);

  React.useEffect(() => {
    let active = true;
    getSystemSettings()
      .then((settings) => {
        if (!active) return;
        const candidates = settings.ai.chat.candidates.filter((c) => c.enabled !== false);
        setOptions([AUTO_OPTION, ...candidates.map(toOption)]);
      })
      .catch(() => {
        if (active) setOptions([AUTO_OPTION]);
      });
    return () => {
      active = false;
    };
  }, []);

  return options;
}
