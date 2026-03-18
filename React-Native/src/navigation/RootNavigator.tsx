import React, { useMemo, useRef } from "react";
import { BackHandler, Platform, ToastAndroid } from "react-native";
import {
  NavigationContainer,
  DefaultTheme as NavigationLightTheme,
  DarkTheme as NavigationDarkTheme,
  NavigationContainerRef
} from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Icon, IconButton, MD3Theme, PaperProvider } from "react-native-paper";

import { RootStackParamList, BottomTabParamList } from "./types";
import { HomeScreen } from "../screens/HomeScreen";
import { CalendarScreen } from "../screens/CalendarScreen";
import { AnalyticsScreen } from "../screens/AnalyticsScreen";
import { SettingsScreen } from "../screens/SettingsScreen";

const RootStack = createNativeStackNavigator<RootStackParamList>();
const Tabs = createBottomTabNavigator<BottomTabParamList>();

function MainTabs(): React.JSX.Element {
  return (
    <Tabs.Navigator
      initialRouteName="Home"
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ color, size }) => {
          const icon =
            route.name === "Home"
              ? "home-variant"
              : route.name === "Calendar"
                ? "calendar-month"
                : "chart-line";
          return <Icon source={icon} color={color} size={size} />;
        },
        tabBarLabelStyle: {
          paddingBottom: 4,
          fontSize: 12
        },
        tabBarStyle: {
          height: 68,
          paddingBottom: 8,
          paddingTop: 8
        },
        lazy: true
      })}
    >
      <Tabs.Screen name="Home" component={HomeScreen} />
      <Tabs.Screen name="Calendar" component={CalendarScreen} />
      <Tabs.Screen name="Analytics" component={AnalyticsScreen} />
    </Tabs.Navigator>
  );
}

export function RootNavigator({ theme }: { theme: MD3Theme }): React.JSX.Element {
  const navigationRef = useRef<NavigationContainerRef<RootStackParamList>>(null);
  const lastBackPress = useRef(0);

  React.useEffect(() => {
    if (Platform.OS !== "android") {
      return;
    }

    const subscription = BackHandler.addEventListener("hardwareBackPress", () => {
      const route = navigationRef.current?.getCurrentRoute()?.name;
      if (route !== "MainTabs") {
        return false;
      }

      const now = Date.now();
      if (now - lastBackPress.current < 1500) {
        BackHandler.exitApp();
        return true;
      }

      lastBackPress.current = now;
      ToastAndroid.show("Press back again to exit", ToastAndroid.SHORT);
      return true;
    });

    return () => subscription.remove();
  }, []);

  const navTheme = useMemo(
    () => ({
      ...(theme.dark ? NavigationDarkTheme : NavigationLightTheme),
      colors: {
        ...(theme.dark ? NavigationDarkTheme.colors : NavigationLightTheme.colors),
        background: theme.colors.background,
        primary: theme.colors.primary,
        card: theme.colors.surface
      }
    }),
    [theme]
  );

  return (
    <PaperProvider theme={theme}>
      <NavigationContainer ref={navigationRef} theme={navTheme}>
        <RootStack.Navigator>
          <RootStack.Screen
            name="MainTabs"
            component={MainTabs}
            options={({ navigation }) => ({
              title: "Mood Tracker",
              headerRight: () => (
                <IconButton icon="cog-outline" size={24} onPress={() => navigation.navigate("Settings")} />
              )
            })}
          />
          <RootStack.Screen name="Settings" component={SettingsScreen} />
        </RootStack.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
