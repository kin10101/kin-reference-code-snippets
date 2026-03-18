import AsyncStorage from "@react-native-async-storage/async-storage";

import { STORAGE_KEYS } from "../constants/storageKeys";
import { AppSettings, MoodEntry } from "../types/models";

const DEFAULT_SETTINGS: AppSettings = {
  theme: "system",
  accentColor: "#4F7CFF",
  notificationsEnabled: true,
  notificationTimes: ["09:00", "12:00", "15:00", "20:00"],
  notificationSoundEnabled: true,
  notificationVibrationEnabled: true
};

export async function loadEntries(): Promise<MoodEntry[]> {
  const raw = await AsyncStorage.getItem(STORAGE_KEYS.ENTRIES);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as MoodEntry[];
    return parsed.sort((a, b) => a.date.localeCompare(b.date));
  } catch {
    return [];
  }
}

export async function saveEntries(entries: MoodEntry[]): Promise<void> {
  await AsyncStorage.setItem(STORAGE_KEYS.ENTRIES, JSON.stringify(entries));
}

export async function loadSettings(): Promise<AppSettings> {
  const raw = await AsyncStorage.getItem(STORAGE_KEYS.SETTINGS);
  if (!raw) {
    return DEFAULT_SETTINGS;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      ...DEFAULT_SETTINGS,
      ...parsed,
      notificationTimes: parsed.notificationTimes?.length ? parsed.notificationTimes : DEFAULT_SETTINGS.notificationTimes
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  await AsyncStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
}

export async function clearAllData(): Promise<void> {
  await AsyncStorage.multiRemove([STORAGE_KEYS.ENTRIES, STORAGE_KEYS.SETTINGS]);
}

export { DEFAULT_SETTINGS };
