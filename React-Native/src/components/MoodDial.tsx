import React, { memo, useMemo } from "react";
import { StyleSheet, View } from "react-native";
import { IconButton, Text, useTheme } from "react-native-paper";
import * as Haptics from "expo-haptics";

import { getMoodColor, getMoodLabel } from "../constants/moodScale";

interface MoodDialProps {
  value: number;
  onChange: (value: number) => void;
}

function MoodDialImpl({ value, onChange }: MoodDialProps): React.JSX.Element {
  const theme = useTheme();
  const moodColor = useMemo(() => getMoodColor(value), [value]);

  const update = (next: number): void => {
    const bounded = Math.min(10, Math.max(1, next));
    if (bounded !== value) {
      onChange(bounded);
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };

  return (
    <View style={styles.container}>
      <View style={[styles.outerRing, { borderColor: moodColor, backgroundColor: theme.colors.elevation.level2 }]}>
        <View style={[styles.innerRing, { backgroundColor: theme.colors.elevation.level1 }]}>
          <Text variant="displayLarge" style={[styles.score, { color: moodColor }]}>
            {value}
          </Text>
          <Text variant="titleMedium" style={styles.titleText}>
            {getMoodLabel(value)}
          </Text>
          <View style={styles.controlsRow}>
            <IconButton
              icon="minus"
              mode="contained-tonal"
              onPress={() => update(value - 1)}
              size={24}
              style={styles.controlButton}
            />
            <IconButton
              icon="plus"
              mode="contained-tonal"
              onPress={() => update(value + 1)}
              size={24}
              style={styles.controlButton}
            />
          </View>
        </View>
      </View>
    </View>
  );
}

export const MoodDial = memo(MoodDialImpl);

const styles = StyleSheet.create({
  container: {
    width: "100%",
    alignItems: "center",
    marginBottom: 18
  },
  outerRing: {
    width: 260,
    height: 260,
    borderRadius: 130,
    borderWidth: 8,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.16,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 10 },
    elevation: 14
  },
  innerRing: {
    width: 220,
    height: 220,
    borderRadius: 110,
    justifyContent: "center",
    alignItems: "center",
    padding: 16,
    shadowColor: "#FFFFFF",
    shadowOpacity: 0.28,
    shadowRadius: 12,
    shadowOffset: { width: -4, height: -4 }
  },
  score: {
    fontWeight: "700"
  },
  titleText: {
    marginTop: 4,
    marginBottom: 14,
    opacity: 0.85
  },
  controlsRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8
  },
  controlButton: {
    minWidth: 56,
    minHeight: 56
  }
});
