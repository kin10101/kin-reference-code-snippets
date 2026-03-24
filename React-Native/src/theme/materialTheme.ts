import {
  MD3DarkTheme,
  MD3LightTheme,
  configureFonts,
  MD3Theme
} from "react-native-paper";

import { DynamicColorSet } from "./dynamicColors";

type ThemeMode = "light" | "dark";

const brandTypography = configureFonts({
  config: {
    displayLarge: { fontFamily: "sans-serif-medium", letterSpacing: -0.6 },
    displayMedium: { fontFamily: "sans-serif-medium", letterSpacing: -0.4 },
    displaySmall: { fontFamily: "sans-serif-medium", letterSpacing: -0.3 },
    headlineLarge: { fontFamily: "sans-serif-medium", letterSpacing: -0.2 },
    headlineMedium: { fontFamily: "sans-serif-medium", letterSpacing: -0.1 },
    headlineSmall: { fontFamily: "sans-serif-medium" },
    titleLarge: { fontFamily: "sans-serif-medium" },
    titleMedium: { fontFamily: "sans-serif-medium" },
    titleSmall: { fontFamily: "sans-serif-medium" },
    bodyLarge: { fontFamily: "sans-serif" },
    bodyMedium: { fontFamily: "sans-serif" },
    bodySmall: { fontFamily: "sans-serif" },
    labelLarge: { fontFamily: "sans-serif-medium" },
    labelMedium: { fontFamily: "sans-serif-medium" },
    labelSmall: { fontFamily: "sans-serif-medium" }
  }
});

export function buildPaperTheme(
  mode: ThemeMode,
  dynamicColors: DynamicColorSet,
  accentColor?: string
): MD3Theme {
  const base = mode === "dark" ? MD3DarkTheme : MD3LightTheme;
  const dynamicPrimary = mode === "dark" ? dynamicColors.darkPrimary : dynamicColors.lightPrimary;
  const primary = accentColor ?? dynamicPrimary;

  return {
    ...base,
    fonts: brandTypography,
    colors: {
      ...base.colors,
      primary: primary ?? base.colors.primary,
      secondaryContainer: mode === "dark" ? "#29333B" : "#DCEEFF",
      elevation: {
        level0: "transparent",
        level1: mode === "dark" ? "#1B1E20" : "#F3F6F9",
        level2: mode === "dark" ? "#212529" : "#E9EEF2",
        level3: mode === "dark" ? "#282D30" : "#E2E8EE",
        level4: mode === "dark" ? "#2E3337" : "#DCE3EA",
        level5: mode === "dark" ? "#333A3F" : "#D4DDE6"
      }
    }
  };
}
