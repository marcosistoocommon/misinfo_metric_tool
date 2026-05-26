import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import Image from "next/image";
import { useMemo, useState } from "react";

import type { Claim } from "@/lib/store";
import { cn, extractDomain } from "@/lib/utils";
import type { ClaimSource } from "@/types";
import { Favicon } from "./ui/favicon";

interface SourcePillsProps {
  verdicts: Claim[];
  maxSources?: number;
}

export const SourcePills = ({ verdicts, maxSources = 7 }: SourcePillsProps) => {
  const [expanded, setExpanded] = useState(false);

  const uniqueSources = useMemo(() => {
    const sources = new Map<string, ClaimSource>();

    for (const verdict of verdicts) {
      for (const source of verdict.sources ?? []) {
        if (!sources.has(source.url)) {
          sources.set(source.url, source);
        }
      }
    }

    return Array.from(sources.values());
  }, [verdicts]);

  if (uniqueSources.length === 0) {
    return null;
  }

  // Determine which sources to display
  const displaySources = expanded
    ? uniqueSources
    : uniqueSources.slice(0, maxSources);

  const hasMoreSources = uniqueSources.length > maxSources;
  const hiddenSourcesCount = uniqueSources.length - maxSources;

  return (
    <div>
      <div className="flex items-center justify-between">
        <h3 className="mb-2.5 font-medium text-sm">Sources</h3>
        {hasMoreSources && (
          <button
            className="flex items-center gap-1 text-neutral-500 text-xs transition-colors hover:text-neutral-900"
            onClick={() => setExpanded(!expanded)}
            type="button"
          >
            {expanded ? (
              <>
                <span>Show less</span>
                <ChevronUp className="h-3 w-3" />
              </>
            ) : (
              <>
                <span>Show all {uniqueSources.length}</span>
                <ChevronDown className="h-3 w-3" />
              </>
            )}
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        {displaySources.map((source) => {
          const domain = extractDomain(source.url);
          const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=16`;
          const title = source.title || domain;

          return (
            <a
              className={cn(
                "flex items-center gap-1.5 px-2 py-0.5 text-xs",
                "rounded-md border border-neutral-200 bg-white text-neutral-700",
                "transition-all duration-150 ease-in-out",
                "hover:border-neutral-300 hover:bg-neutral-100 hover:shadow-sm",
                "focus:outline-none focus:ring-1 focus:ring-neutral-300"
              )}
              href={source.url}
              key={source.url}
              rel="noreferrer"
              target="_blank"
              title={title}
            >
              <Favicon url={source.url} />
              <span className="max-w-40 truncate font-medium">{title}</span>
              <ExternalLink className="h-2.5 w-2.5 flex-shrink-0 text-neutral-400" />
            </a>
          );
        })}

        {!expanded && hasMoreSources && (
          <button
            className={cn(
              "flex items-center gap-1 px-2 py-0.5 text-xs",
              "rounded-md border border-neutral-200 bg-neutral-50 text-neutral-500",
              "hover:border-neutral-300 hover:bg-neutral-100 hover:text-neutral-700",
              "transition-all duration-150 ease-in-out",
              "focus:outline-none focus:ring-1 focus:ring-neutral-300"
            )}
            onClick={() => setExpanded(true)}
            type="button"
          >
            +{hiddenSourcesCount} more
            <ChevronDown className="h-3 w-3" />
          </button>
        )}
      </div>
    </div>
  );
};
