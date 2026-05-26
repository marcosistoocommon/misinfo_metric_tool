export type CheckListItem = {
  id: string;
  slug: string;
  title: string | null;
  status: string;
  createdAt: Date;
  updatedAt: Date;
  completedAt: Date | null;
  textPreview?: string | null;
};

export type CheckGroup = {
  label: string;
  checks: CheckListItem[];
};

export type GroupedChecks = {
  today: CheckListItem[];
  yesterday: CheckListItem[];
  lastSevenDays: CheckListItem[];
  lastThirtyDays: CheckListItem[];
  older: CheckListItem[];
};

const ensureDate = (date: Date | string): Date => {
  return typeof date === "string" ? new Date(date) : date;
};

const getDateBoundaries = (() => {
  let cached: {
    today: Date;
    yesterday: Date;
    sevenDaysAgo: Date;
    thirtyDaysAgo: Date;
    todayStart: Date;
    yesterdayStart: Date;
  } | null = null;

  return () => {
    if (!cached) {
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
      const sevenDaysAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
      const thirtyDaysAgo = new Date(
        today.getTime() - 30 * 24 * 60 * 60 * 1000
      );

      cached = {
        today,
        yesterday,
        sevenDaysAgo,
        thirtyDaysAgo,
        todayStart: today,
        yesterdayStart: yesterday,
      };
    }
    return cached;
  };
})();

const isToday = (date: Date | string): boolean => {
  const dateObj = ensureDate(date);
  const { todayStart } = getDateBoundaries();
  const nextDay = new Date(todayStart.getTime() + 24 * 60 * 60 * 1000);

  return dateObj >= todayStart && dateObj < nextDay;
};

const isYesterday = (date: Date | string): boolean => {
  const dateObj = ensureDate(date);
  const { yesterdayStart, todayStart } = getDateBoundaries();

  return dateObj >= yesterdayStart && dateObj < todayStart;
};

const isWithinDays = (date: Date | string, days: number): boolean => {
  const dateObj = ensureDate(date);
  const boundaries = getDateBoundaries();

  if (days === 7) {
    return dateObj >= boundaries.sevenDaysAgo;
  }
  if (days === 30) {
    return dateObj >= boundaries.thirtyDaysAgo;
  }

  const daysBefore = new Date();
  daysBefore.setDate(daysBefore.getDate() - days);
  return dateObj >= daysBefore;
};

export const groupChecksByTime = (checks: CheckListItem[]): GroupedChecks => {
  const grouped: GroupedChecks = {
    today: [],
    yesterday: [],
    lastSevenDays: [],
    lastThirtyDays: [],
    older: [],
  };

  for (const check of checks) {
    const relevantDate = check.updatedAt;

    if (isToday(relevantDate)) {
      grouped.today.push(check);
    } else if (isYesterday(relevantDate)) {
      grouped.yesterday.push(check);
    } else if (isWithinDays(relevantDate, 7)) {
      grouped.lastSevenDays.push(check);
    } else if (isWithinDays(relevantDate, 30)) {
      grouped.lastThirtyDays.push(check);
    } else {
      grouped.older.push(check);
    }
  }

  return grouped;
};

export const filterChecks = (
  checks: CheckListItem[],
  searchTerm: string
): CheckListItem[] => {
  if (!searchTerm.trim()) return checks;

  const normalizedSearch = searchTerm.toLowerCase().trim();

  return checks.filter((check) =>
    check.title?.toLowerCase().includes(normalizedSearch)
  );
};

export const getCheckGroupsForDisplay = (
  grouped: GroupedChecks
): CheckGroup[] => {
  const groups: CheckGroup[] = [];

  if (grouped.today.length > 0) {
    groups.push({ label: "Today", checks: grouped.today });
  }

  if (grouped.yesterday.length > 0) {
    groups.push({ label: "Yesterday", checks: grouped.yesterday });
  }

  if (grouped.lastSevenDays.length > 0) {
    groups.push({ label: "Last 7 Days", checks: grouped.lastSevenDays });
  }

  if (grouped.lastThirtyDays.length > 0) {
    groups.push({ label: "Last 30 Days", checks: grouped.lastThirtyDays });
  }

  if (grouped.older.length > 0) {
    groups.push({ label: "Older", checks: grouped.older });
  }

  return groups;
};
