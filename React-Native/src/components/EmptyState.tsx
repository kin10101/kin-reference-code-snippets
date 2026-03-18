import React from "react";
import { StyleSheet, View } from "react-native";
import { Text } from "react-native-paper";

interface EmptyStateProps {
  title: string;
  description: string;
}

export function EmptyState({ title, description }: EmptyStateProps): React.JSX.Element {
  return (
    <View style={styles.container}>
      <Text variant="headlineSmall">{title}</Text>
      <Text style={styles.description}>{description}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 40,
    paddingHorizontal: 24
  },
  description: {
    marginTop: 8,
    opacity: 0.7,
    textAlign: "center"
  }
});
