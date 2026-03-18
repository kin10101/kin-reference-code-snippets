import React, { useMemo, useState } from "react";
import { Alert, ScrollView, StyleSheet, View } from "react-native";
import DateTimePicker, { DateTimePickerEvent } from "@react-native-community/datetimepicker";
import {
  Button,
  Divider,
  List,
  RadioButton,
  Switch,
  Text,
  useTheme
} from "react-native-paper";

import { exportAsCsv, exportAsJson, exportAsXlsx } from "../services/exportService";
import { useAppContext } from "../state/AppProvider";
import { formatTimeLabel } from "../utils/date";

const ACCENT_PRESETS = ["#4F7CFF", "#16A34A", "#E11D48", "#D97706", "#0891B2"];

export function SettingsScreen(): React.JSX.Element {
  const theme = useTheme();
  const { settings, updateSettings, entries, deleteAllData } = useAppContext();

  const [timeIndex, setTimeIndex] = useState<number | null>(null);

  const pickerDate = useMemo(() => {
    if (timeIndex === null) {
      return new Date();
    }

    const timeValue = settings.notificationTimes[timeIndex] ?? "09:00";
    const [hour, minute] = timeValue.split(":").map(Number);
    const date = new Date();
    date.setHours(hour ?? 9, minute ?? 0, 0, 0);
    return date;
  }, [settings.notificationTimes, timeIndex]);

  const onNotificationTimeChange = async (event: DateTimePickerEvent, selected?: Date): Promise<void> => {
    if (event.type === "dismissed" || timeIndex === null || !selected) {
      setTimeIndex(null);
      return;
    }

    const formatted = `${String(selected.getHours()).padStart(2, "0")}:${String(selected.getMinutes()).padStart(2, "0")}`;
    const next = [...settings.notificationTimes];
    next[timeIndex] = formatted;
    await updateSettings({ notificationTimes: next });
    setTimeIndex(null);
  };

  const clearData = (): void => {
    Alert.alert("Clear all mood data?", "This action cannot be undone.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: () => {
          void deleteAllData();
        }
      }
    ]);
  };

  return (
    <ScrollView contentContainerStyle={styles.content}>
      <Text variant="headlineSmall">Settings</Text>

      <List.Section>
        <List.Subheader>Theme</List.Subheader>
        <RadioButton.Group
          onValueChange={(value: string) => void updateSettings({ theme: value as "light" | "dark" | "system" })}
          value={settings.theme}
        >
          <RadioButton.Item label="System default" value="system" />
          <RadioButton.Item label="Light" value="light" />
          <RadioButton.Item label="Dark" value="dark" />
        </RadioButton.Group>

        <Text variant="labelLarge" style={styles.sectionLabel}>
          Accent color
        </Text>
        <View style={styles.colorRow}>
          {ACCENT_PRESETS.map((color) => (
            <Button
              key={color}
              mode={settings.accentColor === color ? "contained" : "outlined"}
              onPress={() => void updateSettings({ accentColor: color })}
              style={[styles.colorButton, { backgroundColor: settings.accentColor === color ? color : undefined }]}
            >
              {" "}
            </Button>
          ))}
        </View>
      </List.Section>

      <Divider />

      <List.Section>
        <List.Subheader>Notifications</List.Subheader>
        <List.Item
          title="Enable notifications"
          right={() => (
            <Switch
              value={settings.notificationsEnabled}
              onValueChange={(value) => void updateSettings({ notificationsEnabled: value })}
            />
          )}
        />
        <List.Item
          title="Sound"
          right={() => (
            <Switch
              value={settings.notificationSoundEnabled}
              onValueChange={(value) => void updateSettings({ notificationSoundEnabled: value })}
            />
          )}
        />
        <List.Item
          title="Vibration"
          right={() => (
            <Switch
              value={settings.notificationVibrationEnabled}
              onValueChange={(value) => void updateSettings({ notificationVibrationEnabled: value })}
            />
          )}
        />

        {settings.notificationTimes.map((time, index) => (
          <List.Item
            key={`${time}-${index}`}
            title={`Reminder ${index + 1}`}
            description={formatTimeLabel(time)}
            left={() => <List.Icon icon="clock-outline" />}
            onPress={() => setTimeIndex(index)}
            style={styles.touchTarget}
          />
        ))}
      </List.Section>

      {timeIndex !== null ? (
        <DateTimePicker value={pickerDate} mode="time" is24Hour display="default" onChange={onNotificationTimeChange} />
      ) : null}

      <Divider />

      <List.Section>
        <List.Subheader>Data Management</List.Subheader>
        <View style={styles.exportRow}>
          <Button mode="outlined" onPress={() => void exportAsCsv(entries)} disabled={!entries.length}>
            Export CSV
          </Button>
          <Button mode="outlined" onPress={() => void exportAsJson(entries)} disabled={!entries.length}>
            Export JSON
          </Button>
          <Button mode="outlined" onPress={() => void exportAsXlsx(entries)} disabled={!entries.length}>
            Export XLSX
          </Button>
        </View>

        <Button mode="contained-tonal" buttonColor={theme.colors.errorContainer} textColor={theme.colors.onErrorContainer} onPress={clearData}>
          Clear all data
        </Button>
      </List.Section>

      <Divider />

      <List.Section>
        <List.Subheader>About</List.Subheader>
        <Text>Version 1.0.0</Text>
        <Text style={styles.aboutText}>All mood data stays local on your device. No cloud sync by default.</Text>
      </List.Section>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    padding: 16,
    gap: 12,
    paddingBottom: 40
  },
  sectionLabel: {
    marginTop: 10,
    marginBottom: 8,
    marginLeft: 16
  },
  colorRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 16,
    marginBottom: 8
  },
  colorButton: {
    minWidth: 48,
    minHeight: 48,
    borderRadius: 24
  },
  touchTarget: {
    minHeight: 56,
    justifyContent: "center"
  },
  exportRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 12
  },
  aboutText: {
    marginTop: 6,
    opacity: 0.74
  }
});
