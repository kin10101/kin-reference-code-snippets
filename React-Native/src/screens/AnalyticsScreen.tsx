import React, { useMemo, useState } from "react";
import { Dimensions, ScrollView, StyleSheet, View } from "react-native";
import { LineChart, BarChart } from "react-native-chart-kit";
import { SegmentedButtons, Text, useTheme } from "react-native-paper";

import { EmptyState } from "../components/EmptyState";
import { StatCard } from "../components/StatCard";
import { useAppContext } from "../state/AppProvider";
import { DateRange } from "../types/models";
import { filterEntriesByDays } from "../utils/date";

const RANGE_DAYS: Record<DateRange, number> = {
  "7d": 7,
  "30d": 30,
  "90d": 90,
  "1y": 365
};

const chartWidth = Dimensions.get("window").width - 32;

export function AnalyticsScreen(): React.JSX.Element {
  const theme = useTheme();
  const { entries, stats } = useAppContext();
  const [range, setRange] = useState<DateRange>("30d");

  const filtered = useMemo(() => filterEntriesByDays(entries, RANGE_DAYS[range]), [entries, range]);

  const lineData = useMemo(
    () => ({
      labels: filtered.map((entry) => entry.date.slice(5)),
      datasets: [{ data: filtered.map((entry) => entry.mood) }]
    }),
    [filtered]
  );

  const barData = useMemo(() => {
    const buckets = [1, 2, 3, 4].map((weekIndex) => {
      const subset = filtered.slice((weekIndex - 1) * 7, weekIndex * 7);
      if (!subset.length) {
        return 0;
      }
      return subset.reduce((total, entry) => total + entry.mood, 0) / subset.length;
    });

    return {
      labels: ["W1", "W2", "W3", "W4"],
      datasets: [{ data: buckets }]
    };
  }, [filtered]);

  if (!entries.length) {
    return (
      <View style={styles.container}>
        <EmptyState
          title="No analytics yet"
          description="Log moods for a few days and trend charts will appear here."
        />
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.content}>
      <SegmentedButtons
        value={range}
        onValueChange={(value) => setRange(value as DateRange)}
        buttons={[
          { label: "7d", value: "7d" },
          { label: "30d", value: "30d" },
          { label: "90d", value: "90d" },
          { label: "1y", value: "1y" }
        ]}
      />

      <View style={styles.statsGrid}>
        <StatCard title="Avg mood" value={stats.averageMood.toFixed(1)} />
        <StatCard title="Current streak" value={`${stats.currentStreak}d`} />
        <StatCard title="Best day" value={stats.bestDay?.date ?? "-"} subtitle={stats.bestDay ? `Mood ${stats.bestDay.mood}` : undefined} />
        <StatCard title="Worst day" value={stats.worstDay?.date ?? "-"} subtitle={stats.worstDay ? `Mood ${stats.worstDay.mood}` : undefined} />
      </View>

      <Text variant="titleMedium" style={styles.sectionTitle}>
        Mood trend
      </Text>
      <LineChart
        data={lineData}
        width={chartWidth}
        height={220}
        yAxisSuffix=""
        yAxisInterval={1}
        chartConfig={{
          backgroundGradientFrom: theme.colors.elevation.level1,
          backgroundGradientTo: theme.colors.elevation.level1,
          color: () => theme.colors.primary,
          labelColor: () => theme.colors.onSurface,
          propsForDots: {
            r: "4"
          }
        }}
        bezier
        style={styles.chart}
      />

      <Text variant="titleMedium" style={styles.sectionTitle}>
        Weekly averages
      </Text>
      <BarChart
        data={barData}
        width={chartWidth}
        height={220}
        fromZero
        yAxisLabel=""
        chartConfig={{
          backgroundGradientFrom: theme.colors.elevation.level1,
          backgroundGradientTo: theme.colors.elevation.level1,
          color: () => theme.colors.primary,
          labelColor: () => theme.colors.onSurface
        }}
        style={styles.chart}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center"
  },
  content: {
    padding: 16,
    gap: 12,
    paddingBottom: 30
  },
  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10
  },
  sectionTitle: {
    marginTop: 8
  },
  chart: {
    borderRadius: 18
  }
});
