import * as Notifications from "expo-notifications";

import { AppSettings } from "../types/models";

const CHANNEL_ID = "mood-reminders";

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false
  })
});

export async function initializeNotifications(): Promise<void> {
  await Notifications.setNotificationChannelAsync(CHANNEL_ID, {
    name: "Mood Reminder Channel",
    importance: Notifications.AndroidImportance.HIGH,
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
    vibrationPattern: [0, 250, 200, 250],
    lightColor: "#4F7CFF"
  });

  await Notifications.requestPermissionsAsync({
    android: {
      allowAlert: true,
      allowBadge: true,
      allowSound: true
    }
  });
}

export async function scheduleReminderNotifications(settings: AppSettings): Promise<void> {
  await cancelReminderNotifications();

  if (!settings.notificationsEnabled) {
    return;
  }

  for (const time of settings.notificationTimes) {
    const [hour, minute] = time.split(":").map(Number);

    await Notifications.scheduleNotificationAsync({
      content: {
        title: "Mood check-in",
        body: "How are you feeling right now? Log your mood in under 20 seconds.",
        sound: settings.notificationSoundEnabled,
        vibrate: settings.notificationVibrationEnabled ? [0, 250, 200, 250] : undefined
      },
      trigger: {
        channelId: CHANNEL_ID,
        repeats: true,
        hour: hour ?? 9,
        minute: minute ?? 0,
        type: Notifications.SchedulableTriggerInputTypes.DAILY
      }
    });
  }
}

export async function cancelReminderNotifications(): Promise<void> {
  await Notifications.cancelAllScheduledNotificationsAsync();
}
