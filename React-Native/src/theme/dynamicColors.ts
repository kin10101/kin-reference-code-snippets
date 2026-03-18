import { useMaterial3Theme } from "@pchmn/expo-material3-theme";

export interface DynamicColorSet {
  lightPrimary?: string;
  darkPrimary?: string;
}

export function useDynamicColors(): DynamicColorSet {
  const { theme } = useMaterial3Theme();

  return {
    lightPrimary: theme.light?.primary,
    darkPrimary: theme.dark?.primary
  };
}
