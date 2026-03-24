import {
  differenceInCalendarDays,
  format,
  isSameDay,
  parseISO,
  subDays
} from "date-fns";

import { MoodEntry } from "../types/models";

export function toDateKey(date: Date): string {
  return format(date, "yyyy-MM-dd");
}

export function fromDateKey(value: string): Date {
  return parseISO(value);
}

export function formatReadableDate(date: Date): string {
  return format(date, "EEE, MMM d");
}

export function formatTimeLabel(value: string): string {
  const [hour, minute] = value.split(":").map(Number);
  const date = new Date();
  date.setHours(hour ?? 0, minute ?? 0, 0, 0);
  return format(date, "p");
}

export function filterEntriesByDays(entries: MoodEntry[], days: number): MoodEntry[] {
  const threshold = subDays(new Date(), days - 1);
  return entries.filter((entry) => parseISO(entry.date) >= threshold);
}

export function computeCurrentStreak(entries: MoodEntry[]): number {
  if (entries.length === 0) {
    return 0;
  }

  const dates = entries.map((entry) => parseISO(entry.date)).sort((a, b) => b.getTime() - a.getTime());
  let streak = 0;
  let cursor = new Date();

  if (!dates.some((date) => isSameDay(date, cursor))) {
    cursor = subDays(cursor, 1);
  }

  for (const date of dates) {
    const diff = differenceInCalendarDays(cursor, date);
    if (diff === 0) {
      streak += 1;
      cursor = subDays(cursor, 1);
    }
  }

  return streak;
}
