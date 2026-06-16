/**
 * Mock curriculum data — keyed off the 36 production classes
 * (26 letters + 10 digits) plus a few aspirational words. Backed by localStorage
 * via lib/progress.ts. Backend persistence is out of scope for this redesign.
 */

export interface Lesson {
  id: string;
  title: string;
  description: string;
  signs: string[];
  difficulty: 1 | 2 | 3;
  durationMin: number;
}

export interface Unit {
  id: string;
  title: string;
  subtitle: string;
  lessons: Lesson[];
}

export const CURRICULUM: Unit[] = [
  {
    id: "alphabet",
    title: "ASL alphabet",
    subtitle: "The 26-letter manual alphabet — the foundation of fingerspelling.",
    lessons: [
      {
        id: "alphabet-1",
        title: "Letters A – F",
        description: "Your first six handshapes. Start here.",
        signs: ["a", "b", "c", "d", "e", "f"],
        difficulty: 1,
        durationMin: 4,
      },
      {
        id: "alphabet-2",
        title: "Letters G – L",
        description: "Six more letters, including the tricky G and H.",
        signs: ["g", "h", "i", "j", "k", "l"],
        difficulty: 1,
        durationMin: 5,
      },
      {
        id: "alphabet-3",
        title: "Letters M – R",
        description: "Closed-fist shapes that learners often confuse.",
        signs: ["m", "n", "o", "p", "q", "r"],
        difficulty: 2,
        durationMin: 6,
      },
      {
        id: "alphabet-4",
        title: "Letters S – Z",
        description: "Finish the alphabet and tackle Z's iconic motion.",
        signs: ["s", "t", "u", "v", "w", "x", "y", "z"],
        difficulty: 2,
        durationMin: 7,
      },
    ],
  },
  {
    id: "numbers",
    title: "Numbers 0 – 9",
    subtitle: "Counting from one hand. Essential for dates, times, prices.",
    lessons: [
      {
        id: "numbers-1",
        title: "Zero to four",
        description: "Five basic numbers.",
        signs: ["zero", "one", "two", "three", "four"],
        difficulty: 1,
        durationMin: 4,
      },
      {
        id: "numbers-2",
        title: "Five to nine",
        description: "The other five, including the W-shape six.",
        signs: ["five", "six", "seven", "eight", "nine"],
        difficulty: 1,
        durationMin: 4,
      },
    ],
  },
  {
    id: "greetings",
    title: "Greetings & introductions",
    subtitle: "First words to break the ice — open and close a conversation.",
    lessons: [
      {
        id: "greetings-1",
        title: "Welcome & goodbye",
        description: "Open and close a conversation politely.",
        signs: ["welcome", "bye"],
        difficulty: 1,
        durationMin: 3,
      },
      {
        id: "greetings-2",
        title: "Please & friend",
        description: "Two signs you'll use every day.",
        signs: ["please", "friend"],
        difficulty: 1,
        durationMin: 3,
      },
    ],
  },
  {
    id: "everyday",
    title: "Everyday phrases",
    subtitle: "Common words for class, work, and life.",
    lessons: [
      {
        id: "everyday-1",
        title: "Yes, no, today",
        description: "Three common answers and the time-anchor 'today'.",
        signs: ["yes", "no", "today"],
        difficulty: 1,
        durationMin: 3,
      },
      {
        id: "everyday-2",
        title: "Need & want",
        description: "Two ways to ask for something.",
        signs: ["need", "want"],
        difficulty: 1,
        durationMin: 3,
      },
    ],
  },
  {
    id: "coffee-shop",
    title: "Coffee shop",
    subtitle: "Order a drink without saying a word.",
    lessons: [
      {
        id: "coffee-shop-1",
        title: "Hot or cold",
        description: "The first thing the barista asks.",
        signs: ["hot", "cold"],
        difficulty: 1,
        durationMin: 3,
      },
      {
        id: "coffee-shop-2",
        title: "Coffee, water, more",
        description: "Order one, refill another, ask for more.",
        signs: ["coffee", "water", "more"],
        difficulty: 1,
        durationMin: 4,
      },
    ],
  },
  {
    id: "identity",
    title: "Introducing yourself",
    subtitle: "Tell someone your name and who you are.",
    lessons: [
      {
        id: "identity-1",
        title: "Name & teacher",
        description: "Two signs that come up in every classroom.",
        signs: ["name", "teacher"],
        difficulty: 1,
        durationMin: 3,
      },
      {
        id: "identity-2",
        title: "Deaf & hearing",
        description: "Identifying yourself in a Deaf-hearing conversation.",
        signs: ["deaf", "hearing"],
        difficulty: 2,
        durationMin: 4,
      },
    ],
  },
];

export const ALL_LESSONS: Lesson[] = CURRICULUM.flatMap((u) => u.lessons);

export function getLesson(id: string): Lesson | undefined {
  return ALL_LESSONS.find((l) => l.id === id);
}

export function getUnitForLesson(lessonId: string): Unit | undefined {
  return CURRICULUM.find((u) => u.lessons.some((l) => l.id === lessonId));
}

export function getNextLesson(lessonId: string): Lesson | undefined {
  const idx = ALL_LESSONS.findIndex((l) => l.id === lessonId);
  return idx >= 0 ? ALL_LESSONS[idx + 1] : undefined;
}
