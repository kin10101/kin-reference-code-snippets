import React from "react";
import { View, StyleSheet } from "react-native";
import { Button, Text } from "react-native-paper";

interface ErrorBoundaryState {
  hasError: boolean;
}

export class ErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false
  };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return {
      hasError: true
    };
  }

  handleRetry = (): void => {
    this.setState({ hasError: false });
  };

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <Text variant="titleLarge">Something went wrong.</Text>
          <Text style={styles.caption}>Try again to reload the app shell.</Text>
          <Button mode="contained" onPress={this.handleRetry}>
            Retry
          </Button>
        </View>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 12,
    padding: 24
  },
  caption: {
    opacity: 0.75,
    textAlign: "center"
  }
});
