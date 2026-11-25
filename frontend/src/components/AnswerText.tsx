import type { ReactNode } from "react";
import type { SentenceSupport } from "../types";

interface Props {
  text: string;
  support: SentenceSupport[];
  citationNumbers: Set<number>;
  onCite: (n: number) => void;
}

const INLINE_RE = /(\*\*[^*]+\*\*|\[\d{1,2}\])/g;
const BULLET_RE = /^\s*[-*•]\s+/;
const NUMBERED_RE = /^\s*\d+[.)]\s+/;

/**
 * Renders model output without dangerouslySetInnerHTML: paragraphs, simple
 * lists, **bold**, clickable [n] citations, and a wavy underline on sentences
 * the grounding check could not match to the source they cite.
 */
export function AnswerText({ text, support, citationNumbers, onCite }: Props) {
  const flagged = support.filter((s) => !s.supported && s.sentence.trim());
  const weak = flagged.map((s) => s.sentence.trim());
  const tooltipFor = (sentence: string) => {
    const missing = flagged.find((s) => s.sentence.trim() === sentence)?.missing_terms ?? [];
    return missing.length
      ? `Not found in the cited source: ${missing.join(", ")}. Check the source before relying on this sentence.`
      : "This sentence does not match the source it cites. Check the source before relying on it.";
  };

  const renderInline = (segment: string, keyBase: string): ReactNode[] => {
    const parts = segment.split(INLINE_RE);
    const out: ReactNode[] = [];
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i];
      const key = `${keyBase}-${i}`;
      const cite = /^\[(\d{1,2})\]$/.exec(part);
      if (cite) {
        const n = Number(cite[1]);
        if (!citationNumbers.has(n)) continue;
        // Keep a chip together with the punctuation that follows it, so a
        // full stop never wraps onto a line of its own.
        const next = parts[i + 1] ?? "";
        const trailing = /^[.,;:!?)]+/.exec(next)?.[0] ?? "";
        if (trailing) parts[i + 1] = next.slice(trailing.length);
        // ...and with the word before it, so a chip never starts a line alone.
        let leading = "";
        const previous = out[out.length - 1];
        if (typeof previous === "string") {
          const word = /\S+\s*$/.exec(previous)?.[0] ?? "";
          if (word) {
            out[out.length - 1] = previous.slice(0, previous.length - word.length);
            leading = word;
          }
        }
        out.push(
          <span key={key} className="cite-group">
            {leading}
            <button type="button" className="cite" onClick={() => onCite(n)} aria-label={`Show source ${n}`}>
              {n}
            </button>
            {trailing}
          </span>,
        );
        continue;
      }
      if (part.startsWith("**") && part.endsWith("**")) out.push(<strong key={key}>{part.slice(2, -2)}</strong>);
      else if (part) out.push(part);
    }
    return out;
  };

  const renderWithWeak = (line: string, keyBase: string): ReactNode[] => {
    const out: ReactNode[] = [];
    let rest = line;
    let index = 0;
    while (rest) {
      let hit = -1;
      let sentence = "";
      for (const candidate of weak) {
        const at = rest.indexOf(candidate);
        if (at !== -1 && (hit === -1 || at < hit)) {
          hit = at;
          sentence = candidate;
        }
      }
      if (hit === -1) {
        out.push(...renderInline(rest, `${keyBase}-${index}`));
        break;
      }
      if (hit > 0) out.push(...renderInline(rest.slice(0, hit), `${keyBase}-${index}a`));
      out.push(
        <span
          key={`${keyBase}-${index}w`}
          className="weak"
          title={tooltipFor(sentence)}
        >
          {renderInline(sentence, `${keyBase}-${index}b`)}
        </span>,
      );
      rest = rest.slice(hit + sentence.length);
      index += 1;
    }
    return out;
  };

  const blocks = text.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);

  return (
    <div className="msg-body">
      {blocks.map((block, bi) => {
        const lines = block.split("\n").filter((l) => l.trim());
        if (lines.every((l) => BULLET_RE.test(l))) {
          return (
            <ul key={bi}>
              {lines.map((l, li) => (
                <li key={li}>{renderWithWeak(l.replace(BULLET_RE, ""), `${bi}-${li}`)}</li>
              ))}
            </ul>
          );
        }
        if (lines.every((l) => NUMBERED_RE.test(l))) {
          return (
            <ol key={bi}>
              {lines.map((l, li) => (
                <li key={li}>{renderWithWeak(l.replace(NUMBERED_RE, ""), `${bi}-${li}`)}</li>
              ))}
            </ol>
          );
        }
        return <p key={bi}>{renderWithWeak(lines.join(" "), `${bi}`)}</p>;
      })}
    </div>
  );
}
