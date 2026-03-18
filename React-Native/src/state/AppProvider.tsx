import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import { useColorScheme } from "react-native";

import { useDynamicColors } from "../theme/dynamicColors";
import { MoodEntry, AppSettings, MoodStat } from "../types/models";
import {
  clearAllData,
  DEFAULT_SETTINGS,
  loadEntries,
  loadSettings,
  saveEntries,
  saveSettings
} from "../services/storage";
import {
  initializeNotifications,
  scheduleReminderNotifications
} from "../services/notifications";
import { computeCurrentStreak, toDateKey } from "../utils/date";

interface AppContextValue {
  entries: MoodEntry[];
  settings: AppSettings;
  resolvedTheme: "light" | "dark";
  dynamicColors: ReturnType<typeof useDynamicColors>;
  isBootstrapping: boolean;
  upsertMoodEntry: (entry: Omit<MoodEntry, "id" | "timestamp">) => Promise<void>;
  getEntryByDate: (date: string) => MoodEntry | undefined;
  updateSettings: (next: Partial<AppSettings>) => Promise<void>;
  deleteAllData: () => Promise<void>;
  stats: MoodStat;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: React.ReactNode }): React.JSX.Element {
  const colorScheme = useColorScheme();
  const dynamicColors = useDynamicColors();
  const [entries, setEntries] = useState<MoodEntry[]>([]);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [isBootstrapping, setIsBootstrapping] = useState(true);

  useEffect(() => {
    let isMounted = true;

    async function bootstrap(): Promise<void> {
      await initializeNotifications();
      const [storedEntries, storedSettings] = await Promise.all([loadEntries(), loadSettings()]);

      if (!isMounted) {
        return;
      }

      setEntries(storedEntries);
      setSettings(storedSettings);
      setIsBootstrapping(false);
      await scheduleReminderNotifications(storedSettings);
    }

    void bootstrap();

    return () => {
      isMounted = false;
    };
  }, []);

  const resolvedTheme = settings.theme === "system" ? (colorScheme === "dark" ? "dark" : "light") : settings.theme;

  const upsertMoodEntry = useCallback(
    async (payload: Omit<MoodEntry, "id" | "timestamp">) => {
      setEntries((prev) => {
        const existing = prev.find((entry) => entry.date === payload.date);
        const nextEntry: MoodEntry = {
          id: existing?.id ?? `${payload.date}-${Math.random().toString(36).slice(2, 8)}`,
          date: payload.date,
          mood: payload.mood,
          notes: payload.notes?.trim() ? payload.notes.trim() : undefined,
          timestamp: Date.now()
        };

        const next = existing
          ? prev.map((entry) => (entry.date === payload.date ? nextEntry : entry))
          : [...prev, nextEntry];

        const sorted = next.sort((a, b) => a.date.localeCompare(b.date));
        void saveEntries(sorted);
        return sorted;
      });
    },
    []
  );

  const getEntryByDate = useCallback(
    (date: string) => entries.find((entry) => entry.date === date),
    [entries]
  );

  const updateSettings = useCallback(
    async (next: Partial<AppSettings>) => {
      const merged = {
        ...settings,
        ...next
      };
      setSettings(merged);
      await saveSettings(merged);
      await scheduleReminderNotifications(merged);
    },
    [settings]
  );

  const deleteAllData = useCallback(async () => {
    await clearAllData();
    setEntries([]);
    setSettings(DEFAULT_SETTINGS);
    await saveSettings(DEFAULT_SETTINGS);
    await scheduleReminderNotifications(DEFAULT_SETTINGS);
  }, []);

  const stats = useMemo<MoodStat>(() => {
    if (entries.length === 0) {
      return {
        averageMood: 0,
        currentStreak: 0
      };
    }

    const averageMood = entries.reduce((total, entry) => total + entry.mood, 0) / entries.length;
    const sorted = [...entries].sort((a, b) => a.mood - b.mood);

    return {
      averageMood,
      bestDay: sorted.at(-1),
      worstDay: sorted.at(0),
      currentStreak: computeCurrentStreak(entries)
    };
  }, [entries]);

  const value = useMemo<AppContextValue>(
    () => ({
      entries,
      settings,
      resolvedTheme,
      dynamicColors,
      isBootstrapping,
      upsertMoodEntry,
      getEntryByDate,
      updateSettings,
      deleteAllData,
      stats
    }),
    [
      entries,
      settings,
      resolvedTheme,
      dynamicColors,
      isBootstrapping,
      upsertMoodEntry,
      getEntryByDate,
      updateSettings,
      deleteAllData,
      stats
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext(): AppContextValue {
  const value = useContext(AppContext);
  if (!value) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return value;
}

export function useTodayEntry(): MoodEntry | undefined {
  const { getEntryByDate } = useAppContext();
  return getEntryByDate(toDateKey(new Date()));
}
