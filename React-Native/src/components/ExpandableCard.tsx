import React from "react";
import { StyleSheet, View } from "react-native";
import { Card, IconButton, Text } from "react-native-paper";

interface ExpandableCardProps {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

export function ExpandableCard({ title, expanded, onToggle, children }: ExpandableCardProps): React.JSX.Element {
  return (
    <Card style={styles.card} mode="elevated">
      <Card.Title
        title={title}
        right={() => (
          <IconButton
            icon={expanded ? "chevron-up" : "chevron-down"}
            onPress={onToggle}
            style={styles.toggleButton}
          />
        )}
      />
      {expanded ? <View style={styles.content}>{children}</View> : <Text style={styles.collapsedText}>Tap to add notes</Text>}
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 20,
    marginBottom: 18
  },
  toggleButton: {
    marginRight: 8,
    minWidth: 48,
    minHeight: 48
  },
  content: {
    paddingHorizontal: 16,
    paddingBottom: 16
  },
  collapsedText: {
    paddingHorizontal: 16,
    paddingBottom: 16,
    opacity: 0.65
  }
});
