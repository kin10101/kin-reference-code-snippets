# Android Mood Tracker (React Native + TypeScript)

This is a learning-focused, Android-first React Native reference implementation of a mood tracking app with Material Design 3 styling, offline persistence, local notifications, bottom navigation, and analytics.

## What this implementation includes

- Daily mood logging on a `1-10` scale with optional notes.
- Offline-first local persistence with `AsyncStorage`.
- Android notification channel + scheduled reminders at 4 configurable times.
- Fast home interaction flow (mood dial + optional notes + FAB save).
- Bottom tabs:
	- Home
	- Calendar (week / month / year modes)
	- Analytics (line chart, bar chart, stats cards)
- Settings screen:
	- Theme mode (`light` / `dark` / `system`)
	- Accent color presets
	- Notification toggles + custom times + sound/vibration
	- Data export (`CSV`, `JSON`, `XLSX`) via share sheet
	- Clear all data confirmation
- Error boundary and startup loading state.
- Android back button handling with double-press exit on root tabs.

## Tech stack

- `expo` + `react-native` + `typescript`
- `@react-navigation/native` + bottom tabs + native stack
- `react-native-paper` (Material 3)
- `@react-native-async-storage/async-storage`
- `expo-notifications` (Android notification channels + schedules)
- `react-native-calendars`
- `react-native-chart-kit`
- `@gorhom/bottom-sheet`
- `expo-haptics`
- `react-native-share` + `xlsx` + `expo-file-system`

## Project structure

```
React-Native/
	App.tsx
	app.json
	package.json
	tsconfig.json
	src/
		components/
			EmptyState.tsx
			ErrorBoundary.tsx
			ExpandableCard.tsx
			MoodDial.tsx
			StatCard.tsx
		constants/
			moodScale.ts
			storageKeys.ts
		navigation/
			RootNavigator.tsx
			types.ts
		screens/
			AnalyticsScreen.tsx
			CalendarScreen.tsx
			HomeScreen.tsx
			SettingsScreen.tsx
		services/
			exportService.ts
			notifications.ts
			storage.ts
		state/
			AppProvider.tsx
		theme/
			dynamicColors.ts
			materialTheme.ts
		types/
			models.ts
		utils/
			date.ts
```

## Data models

```ts
interface MoodEntry {
	id: string;
	date: string; // YYYY-MM-DD
	mood: number; // 1-10
	notes?: string;
	timestamp: number;
}

interface AppSettings {
	theme: "light" | "dark" | "system";
	accentColor: string;
	notificationsEnabled: boolean;
	notificationTimes: string[]; // ["09:00", "12:00", "15:00", "20:00"]
	notificationSoundEnabled: boolean;
	notificationVibrationEnabled: boolean;
}
```

## Getting started

1. Install dependencies:

```bash
cd React-Native
npm install
```

2. Start the app:

```bash
npm run start
```

3. Run on Android:

```bash
npm run android
```

## Android notes

- Notification reminders are created in channel `mood-reminders`.
- App requests notification permission at startup.
- `app.json` includes Android permissions and adaptive icon config.
- Status bar style follows light/dark theme.
- Hardware back button on root tabs requires double press to exit.

## Performance choices

- Bottom tabs use `lazy` loading.
- Derived values use memoization (`useMemo`).
- Reusable visual components are isolated and light.
- Data lives in memory after initial load and syncs to storage on writes.

## How to extend this app

- Replace AsyncStorage with SQLite if you need very large local datasets.
- Add import flow for `CSV/JSON/XLSX` to complement exports.
- Add onboarding screens and tooltips for first run.
- Add Android home-screen widget in a native module (future enhancement).
- Add tests with React Native Testing Library and Detox.

## Learning tips

- Start with `src/state/AppProvider.tsx` to understand app state flow.
- Then inspect each screen for UI-specific logic.
- Keep services pure (`storage`, `notifications`, `exportService`) so they are easy to test and replace.

