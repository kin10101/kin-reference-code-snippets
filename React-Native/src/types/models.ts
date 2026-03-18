export interface MoodEntry {
  id: string;
  date: string;
  mood: number;
  notes?: string;
  timestamp: number;
}

export interface AppSettings {
  theme: "light" | "dark" | "system";
  accentColor: string;
  notificationsEnabled: boolean;
  notificationTimes: string[];
  notificationSoundEnabled: boolean;
  notificationVibrationEnabled: boolean;
}

export type DateRange = "7d" | "30d" | "90d" | "1y";

export interface MoodStat {
  averageMood: number;
  bestDay?: MoodEntry;
  worstDay?: MoodEntry;
  currentStreak: number;
}
