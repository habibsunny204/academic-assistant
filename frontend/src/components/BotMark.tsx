import { GraduationCap } from "lucide-react";

/** The assistant's mark. A generic emblem, not the university's crest. */
export function BotMark({ size = 22 }: { size?: number }) {
  return <GraduationCap size={size} strokeWidth={2} />;
}
