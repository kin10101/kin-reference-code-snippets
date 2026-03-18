import React, { useEffect, useMemo, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { Chip, FAB, Snackbar, Text, TextInput, useTheme } from "react-native-paper";

import { MoodDial } from "../components/MoodDial";
import { ExpandableCard } from "../components/ExpandableCard";
import { useAppContext } from "../state/AppProvider";
import { formatReadableDate, toDateKey } from "../utils/date";
import { getMoodLabel } from "../constants/moodScale";

export function HomeScreen(): React.JSX.Element {
  const theme = useTheme();
  const { getEntryByDate, upsertMoodEntry } = useAppContext();

  const today = useMemo(() => new Date(), []);
  const dateKey = useMemo(() => toDateKey(today), [today]);
  const existing = getEntryByDate(dateKey);

  const [mood, setMood] = useState(existing?.mood ?? 6);
  const [notes, setNotes] = useState(existing?.notes ?? "");
  const [notesOpen, setNotesOpen] = useState(Boolean(existing?.notes));
  const [snackbar, setSnackbar] = useState("");

  useEffect(() => {
    setMood(existing?.mood ?? 6);
    setNotes(existing?.notes ?? "");
  }, [existing?.mood, existing?.notes]);

  const saveEntry = async (): Promise<void> => {
    await upsertMoodEntry({
      date: dateKey,
      mood,
      notes
    });
    setSnackbar(existing ? "Mood updated" : "Mood saved");
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={[styles.heroCard, { backgroundColor: theme.colors.elevation.level2 }]}> 
          <Chip icon="calendar-today" style={styles.dateBadge}>
            {formatReadableDate(today)}
          </Chip>
          <Text variant="headlineSmall">How are you feeling?</Text>
          <Text style={styles.subtitle}>Score {mood}: {getMoodLabel(mood)}</Text>
          <MoodDial value={mood} onChange={setMood} />
        </View>

        <ExpandableCard title="Optional notes" expanded={notesOpen} onToggle={() => setNotesOpen((prev) => !prev)}>
          <TextInput
            mode="outlined"
            placeholder="What influenced your mood today?"
            value={notes}
            onChangeText={setNotes}
            multiline
            numberOfLines={4}
          />
        </ExpandableCard>
      </ScrollView>

      <FAB icon="content-save" label="Save" style={styles.fab} onPress={saveEntry} />

      <Snackbar visible={Boolean(snackbar)} onDismiss={() => setSnackbar("")} duration={1800}>
        {snackbar}
      </Snackbar>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  content: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 120
  },
  heroCard: {
    borderRadius: 24,
    padding: 16,
    marginBottom: 14,
    shadowColor: "#000",
    shadowOpacity: 0.16,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 8 },
    elevation: 10
  },
  dateBadge: {
    alignSelf: "flex-start",
    marginBottom: 8
  },
  subtitle: {
    marginTop: 4,
    marginBottom: 10,
    opacity: 0.76
  },
  fab: {
    position: "absolute",
    right: 16,
    bottom: 22,
    minHeight: 56
  }
});
