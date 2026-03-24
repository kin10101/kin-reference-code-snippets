import * as FileSystem from "expo-file-system";
import Share from "react-native-share";
import * as XLSX from "xlsx";

import { MoodEntry } from "../types/models";

function entriesToRows(entries: MoodEntry[]): Array<Record<string, string | number>> {
  return entries.map((entry) => ({
    id: entry.id,
    date: entry.date,
    mood: entry.mood,
    notes: entry.notes ?? "",
    timestamp: entry.timestamp
  }));
}

async function shareFile(fileName: string, data: string, mimeType: string): Promise<void> {
  const path = `${FileSystem.cacheDirectory}${fileName}`;
  await FileSystem.writeAsStringAsync(path, data, {
    encoding: FileSystem.EncodingType.UTF8
  });

  await Share.open({
    url: path,
    type: mimeType,
    failOnCancel: false
  });
}

export async function exportAsJson(entries: MoodEntry[]): Promise<void> {
  await shareFile("mood-tracker-export.json", JSON.stringify(entries, null, 2), "application/json");
}

export async function exportAsCsv(entries: MoodEntry[]): Promise<void> {
  const rows = entriesToRows(entries);
  const headers = ["id", "date", "mood", "notes", "timestamp"];
  const body = rows
    .map((row) =>
      headers
        .map((header) => {
          const value = String(row[header] ?? "").replaceAll('"', '""');
          return `"${value}"`;
        })
        .join(",")
    )
    .join("\n");

  const csv = `${headers.join(",")}\n${body}`;
  await shareFile("mood-tracker-export.csv", csv, "text/csv");
}

export async function exportAsXlsx(entries: MoodEntry[]): Promise<void> {
  const workbook = XLSX.utils.book_new();
  const worksheet = XLSX.utils.json_to_sheet(entriesToRows(entries));
  XLSX.utils.book_append_sheet(workbook, worksheet, "Moods");

  const base64 = XLSX.write(workbook, { type: "base64", bookType: "xlsx" });
  const path = `${FileSystem.cacheDirectory}mood-tracker-export.xlsx`;
  await FileSystem.writeAsStringAsync(path, base64, {
    encoding: FileSystem.EncodingType.Base64
  });

  await Share.open({
    url: path,
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    failOnCancel: false
  });
}
