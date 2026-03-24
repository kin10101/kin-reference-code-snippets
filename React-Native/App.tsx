import "react-native-gesture-handler";
import React from "react";
import { View } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import { ActivityIndicator } from "react-native-paper";

import { AppProvider, useAppContext } from "./src/state/AppProvider";
import { RootNavigator } from "./src/navigation/RootNavigator";
import { buildPaperTheme } from "./src/theme/materialTheme";
import { ErrorBoundary } from "./src/components/ErrorBoundary";

function AppShell(): React.JSX.Element {
  const { resolvedTheme, dynamicColors, settings, isBootstrapping } = useAppContext();
  const paperTheme = buildPaperTheme(resolvedTheme, dynamicColors, settings.accentColor);

  if (isBootstrapping) {
    return (
      <SafeAreaProvider>
        <GestureHandlerRootView style={{ flex: 1 }}>
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
            <ActivityIndicator size="large" />
          </View>
          <StatusBar style={resolvedTheme === "dark" ? "light" : "dark"} />
        </GestureHandlerRootView>
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <ErrorBoundary>
          <RootNavigator theme={paperTheme} />
          <StatusBar style={resolvedTheme === "dark" ? "light" : "dark"} />
        </ErrorBoundary>
      </GestureHandlerRootView>
    </SafeAreaProvider>
  );
}

export default function App(): React.JSX.Element {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}
