import React, { useMemo, useRef, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import BottomSheet from "@gorhom/bottom-sheet";
import { Calendar, DateData } from "react-native-calendars";
import { MarkedDates } from "react-native-calendars/src/types";
import { eachDayOfInterval, endOfWeek, format, parseISO, startOfWeek } from "date-fns";
import {
  Button,
  Chip,
  SegmentedButtons,
  Text,
  TextInput,
  useTheme
} from "react-native-paper";

import { EmptyState } from "../components/EmptyState";
import { getMoodColor } from "../constants/moodScale";
import { useAppContext } from "../state/AppProvider";
import { toDateKey } from "../utils/date";

type ViewMode = "week" | "month" | "year";

export function CalendarScreen(): React.JSX.Element {
  const theme = useTheme();
  const { entries, getEntryByDate, upsertMoodEntry } = useAppContext();
  const bottomSheetRef = useRef<BottomSheet>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("month");
  const [selectedDate, setSelectedDate] = useState(toDateKey(new Date()));
  const selectedEntry = getEntryByDate(selectedDate);
  const [mood, setMood] = useState(selectedEntry?.mood ?? 6);
  const [notes, setNotes] = useState(selectedEntry?.notes ?? "");

  const selectedDateObject = useMemo(() => parseISO(selectedDate), [selectedDate]);

  const markedDates = useMemo<MarkedDates>(() => {
    return entries.reduce<MarkedDates>((acc, entry) => {
      acc[entry.date] = {
        selected: true,
        selectedColor: getMoodColor(entry.mood)
      };
      return acc;
    }, {});
  }, [entries]);

  const onSelectDay = (day: DateData): void => {
    setSelectedDate(day.dateString);
    const dayEntry = getEntryByDate(day.dateString);
    setMood(dayEntry?.mood ?? 6);
    setNotes(dayEntry?.notes ?? "");
    bottomSheetRef.current?.expand();
  };

  const selectDate = (dateString: string): void => {
    setSelectedDate(dateString);
    const dayEntry = getEntryByDate(dateString);
    setMood(dayEntry?.mood ?? 6);
    setNotes(dayEntry?.notes ?? "");
    bottomSheetRef.current?.expand();
  };

  const weekDays = useMemo(
    () =>
      eachDayOfInterval({
        start: startOfWeek(selectedDateObject, { weekStartsOn: 1 }),
        end: endOfWeek(selectedDateObject, { weekStartsOn: 1 })
      }),
    [selectedDateObject]
  );

  const yearlySummary = useMemo(() => {
    return Array.from({ length: 12 }, (_, monthIndex) => {
      const monthlyEntries = entries.filter((entry) => parseISO(entry.date).getMonth() === monthIndex);
      const average = monthlyEntries.length
        ? monthlyEntries.reduce((sum, entry) => sum + entry.mood, 0) / monthlyEntries.length
        : undefined;

      return {
        label: format(new Date(new Date().getFullYear(), monthIndex, 1), "MMM"),
        average,
        count: monthlyEntries.length
      };
    });
  }, [entries]);

  const saveDay = async (): Promise<void> => {
    await upsertMoodEntry({
      date: selectedDate,
      mood,
      notes
    });
    bottomSheetRef.current?.close();
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <SegmentedButtons
          value={viewMode}
          onValueChange={(value: string) => setViewMode(value as ViewMode)}
          buttons={[
            { label: "Week", value: "week" },
            { label: "Month", value: "month" },
            { label: "Year", value: "year" }
          ]}
        />

        {viewMode === "month" ? (
          <View style={[styles.calendarCard, { backgroundColor: theme.colors.elevation.level2 }]}>
            <Calendar markedDates={markedDates} onDayPress={onSelectDay} enableSwipeMonths />
            <Text style={styles.caption}>
              Heat map: Red 1-3, Orange 4-5, Yellow 6-7, Green 8-10
            </Text>
          </View>
        ) : null}

        {viewMode === "week" ? (
          <View style={[styles.calendarCard, { backgroundColor: theme.colors.elevation.level2 }]}>
            <Text variant="titleMedium" style={styles.weekTitle}>
              Week of {format(startOfWeek(selectedDateObject, { weekStartsOn: 1 }), "MMM d")}
            </Text>
            <View style={styles.weekRow}>
              {weekDays.map((day) => {
                const key = format(day, "yyyy-MM-dd");
                const item = getEntryByDate(key);
                return (
                  <Chip
                    key={key}
                    mode={selectedDate === key ? "flat" : "outlined"}
                    onPress={() => selectDate(key)}
                    style={{ backgroundColor: item ? getMoodColor(item.mood) : undefined }}
                  >
                    {format(day, "EEE d")}
                  </Chip>
                );
              })}
            </View>
          </View>
        ) : null}

        {viewMode === "year" ? (
          <View style={[styles.calendarCard, { backgroundColor: theme.colors.elevation.level2 }]}>
            <Text variant="titleMedium" style={styles.weekTitle}>
              Year overview
            </Text>
            <View style={styles.yearGrid}>
              {yearlySummary.map((month) => (
                <View key={month.label} style={styles.yearCell}>
                  <Text variant="labelLarge">{month.label}</Text>
                  <Text>{month.average ? month.average.toFixed(1) : "-"}</Text>
                  <Text style={styles.caption}>{month.count} entries</Text>
                </View>
              ))}
            </View>
          </View>
        ) : null}

        {entries.length === 0 ? (
          <EmptyState
            title="No mood entries yet"
            description="Tap any day to log a mood. Your color heat map will appear automatically."
          />
        ) : null}
      </ScrollView>

      <BottomSheet ref={bottomSheetRef} index={-1} snapPoints={["58%"]}>
        <View style={styles.sheetContent}>
          <Text variant="titleLarge">{selectedDate}</Text>
          <TextInput
            mode="outlined"
            label="Mood score (1-10)"
            value={String(mood)}
            onChangeText={(text: string) => {
              const next = Number(text);
              if (Number.isFinite(next)) {
                setMood(Math.min(10, Math.max(1, next)));
              }
            }}
            keyboardType="number-pad"
          />
          <TextInput
            mode="outlined"
            label="Notes"
            value={notes}
            onChangeText={setNotes}
            multiline
            numberOfLines={4}
          />

          <View style={styles.sheetActions}>
            <Button mode="text" onPress={() => bottomSheetRef.current?.close()}>
              Cancel
            </Button>
            <Button mode="contained" onPress={saveDay}>
              Save
            </Button>
          </View>
        </View>
      </BottomSheet>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  content: {
    padding: 16,
    gap: 14
  },
  calendarCard: {
    borderRadius: 18,
    padding: 12,
    overflow: "hidden"
  },
  caption: {
    marginTop: 10,
    opacity: 0.72,
    paddingHorizontal: 8
  },
  weekTitle: {
    marginBottom: 10
  },
  weekRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8
  },
  yearGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8
  },
  yearCell: {
    width: "31%",
    minHeight: 76,
    borderRadius: 14,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.08)",
    padding: 8
  },
  sheetContent: {
    flex: 1,
    paddingHorizontal: 16,
    gap: 12
  },
  sheetActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
    marginTop: 8
  }
});
