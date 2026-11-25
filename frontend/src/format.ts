const DATE_FMT = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "long", day: "numeric" });
const TIME_FMT = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const DATETIME_FMT = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

function parse(value: string | null | undefined): Date | null {
  if (!value) return null;
  // A bare YYYY-MM-DD is a calendar date, not a UTC instant: parse it as local
  // so "2026-09-12" never renders as 11 September west of UTC.
  const plain = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  const date = plain ? new Date(+plain[1], +plain[2] - 1, +plain[3]) : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export const formatDate = (value: string | null | undefined) => {
  const d = parse(value);
  return d ? DATE_FMT.format(d) : "";
};

export const formatTime = (value: string | null | undefined) => {
  const d = parse(value);
  return d ? TIME_FMT.format(d) : "";
};

export const formatDateTime = (value: string | null | undefined) => {
  const d = parse(value);
  return d ? DATETIME_FMT.format(d) : "";
};

export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 10_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  return n.toLocaleString();
}

// Each category keeps one stripe colour everywhere it appears. Unlisted
// categories share the neutral slot rather than borrowing another's colour.
const CATEGORY_SLOT: Record<string, number> = {
  Registration: 1,
  Examination: 2,
  "Class Routine": 3,
  Curriculum: 4,
  Notices: 5,
  Policies: 6,
  Admission: 7,
};

export const categoryStripe = (category: string) => `var(--cat-${CATEGORY_SLOT[category] ?? 8})`;

/** Client-side id. crypto.randomUUID exists only in secure contexts, so a
 *  UI opened over plain http on a LAN address needs the fallback. */
export function uid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
