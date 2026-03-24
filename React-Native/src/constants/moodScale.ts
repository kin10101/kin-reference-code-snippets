export interface MoodScaleItem {
  score: number;
  title: string;
  description: string[];
  color: string;
}

export const MOOD_SCALE: MoodScaleItem[] = [
  {
    score: 1,
    title: "Severe low",
    description: [
      "You struggle to function.",
      "You avoid people.",
      "Basic tasks feel heavy.",
      "Sleep and appetite are disrupted."
    ],
    color: "#EF4444"
  },
  {
    score: 2,
    title: "Very low",
    description: [
      "You feel persistent sadness or emptiness.",
      "You do what is required, nothing more.",
      "Energy is minimal."
    ],
    color: "#EF4444"
  },
  {
    score: 3,
    title: "Low",
    description: [
      "You feel down most of the day.",
      "Focus is weak.",
      "You cancel optional plans.",
      "You move slowly."
    ],
    color: "#EF4444"
  },
  {
    score: 4,
    title: "Below average",
    description: [
      "You feel off.",
      "Motivation is low but you still function.",
      "Work gets done with effort."
    ],
    color: "#F97316"
  },
  {
    score: 5,
    title: "Neutral",
    description: [
      "No strong positive or negative emotion.",
      "You function normally.",
      "Energy is steady.",
      "No strong drive."
    ],
    color: "#F97316"
  },
  {
    score: 6,
    title: "Slightly positive",
    description: [
      "You feel okay.",
      "You handle tasks without resistance.",
      "You may enjoy small moments."
    ],
    color: "#EAB308"
  },
  {
    score: 7,
    title: "Good",
    description: [
      "You feel engaged.",
      "You start tasks easily.",
      "You are social if given the chance.",
      "Energy is solid."
    ],
    color: "#EAB308"
  },
  {
    score: 8,
    title: "Very good",
    description: [
      "You feel confident and productive.",
      "You seek interaction.",
      "You make progress on important work."
    ],
    color: "#22C55E"
  },
  {
    score: 9,
    title: "Excellent",
    description: [
      "You feel energized and optimistic.",
      "You take initiative.",
      "You feel aligned with your goals."
    ],
    color: "#22C55E"
  },
  {
    score: 10,
    title: "Exceptional",
    description: [
      "You feel peak joy, clarity, and drive.",
      "You perform at your best.",
      "The day feels meaningful and memorable."
    ],
    color: "#22C55E"
  }
];

export function getMoodColor(mood: number): string {
  if (mood <= 3) return "#EF4444";
  if (mood <= 5) return "#F97316";
  if (mood <= 7) return "#EAB308";
  return "#22C55E";
}

export function getMoodLabel(mood: number): string {
  return MOOD_SCALE.find((item) => item.score === mood)?.title ?? "Unknown";
}
