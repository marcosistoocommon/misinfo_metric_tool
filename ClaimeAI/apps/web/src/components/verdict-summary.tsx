import { motion } from "framer-motion";
import { ChevronDown, Link as LinkIcon, MoreHorizontal } from "lucide-react";
import { memo, useState } from "react";
import { Favicon } from "@/components/ui/favicon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { VerdictBadge } from "@/components/ui/verdict-badge";
import type { Claim } from "@/lib/store";
import { extractDomain } from "@/lib/utils";
import type { ClaimSource } from "@/types";

interface VerdictSummaryProps {
  verdicts: Claim[];
  isLoading: boolean;
}

const MAX_VISIBLE_SOURCES = 3;
const INITIAL_CARDS_TO_SHOW = 3;

type SourceLinkProps = {
  source: ClaimSource;
};

const SourceLink = ({ source }: SourceLinkProps) => (
  <a
    aria-label={`View source: ${source.title || source.url}`}
    className="flex items-center rounded-sm border border-neutral-200 p-1 hover:border-neutral-300 hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
    href={source.url}
    rel="noopener noreferrer"
    target="_blank"
    title={source.title || source.url}
  >
    <Favicon url={source.url} />
  </a>
);

type AdditionalSourcesPopoverProps = {
  sources: ClaimSource[];
};

const AdditionalSourcesPopover = ({
  sources,
}: AdditionalSourcesPopoverProps) => (
  <Popover>
    <PopoverTrigger asChild>
      <button
        aria-label={`Show ${sources.length} more sources`}
        className="flex h-6 w-6 items-center justify-center rounded-sm border border-neutral-200 bg-neutral-50 text-neutral-500 hover:border-neutral-300 hover:bg-neutral-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
        type="button"
      >
        <MoreHorizontal className="h-3 w-3" />
      </button>
    </PopoverTrigger>
    <PopoverContent align="end" className="w-auto max-w-xs p-2" side="top">
      <div className="space-y-1.5">
        <p className="font-medium text-neutral-600 text-xs">
          Additional Sources
        </p>
        {sources.map((source) => (
          <a
            className="flex items-center gap-2 rounded-md p-1.5 text-neutral-700 text-xs hover:bg-neutral-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
            href={source.url}
            key={source.url}
            rel="noopener noreferrer"
            target="_blank"
            title={source.title || source.url}
          >
            <Favicon url={source.url} />
            <span className="truncate">
              {source.title || extractDomain(source.url)}
            </span>
            <LinkIcon className="ml-auto h-3 w-3 flex-shrink-0 text-neutral-400" />
          </a>
        ))}
      </div>
    </PopoverContent>
  </Popover>
);

type ToggleButtonProps = {
  show: boolean;
  onClick: () => void;
  remainingCount: number;
};

const ToggleButton = ({ show, onClick, remainingCount }: ToggleButtonProps) => (
  <div className="flex justify-start">
    <button
      className="inline-flex items-center gap-1.5 rounded-md border border-neutral-200 bg-neutral-50 px-3 py-1.5 text-neutral-600 text-xs hover:border-neutral-300 hover:bg-neutral-100"
      onClick={onClick}
      type="button"
    >
      <span>{show ? "Show less" : `Show ${remainingCount} more`}</span>
      <ChevronDown className={`h-3 w-3 ${show ? "rotate-180" : ""}`} />
    </button>
  </div>
);

export const VerdictSummary = memo(function VerdictSummary({
  verdicts,
  isLoading,
}: VerdictSummaryProps) {
  const [showAllCards, setShowAllCards] = useState(false);

  if (verdicts.length === 0) return null;

  const visibleVerdicts = showAllCards
    ? verdicts
    : verdicts.slice(0, INITIAL_CARDS_TO_SHOW);

  const remainingCount = verdicts.length - INITIAL_CARDS_TO_SHOW;
  const hasMoreCards = verdicts.length > INITIAL_CARDS_TO_SHOW;

  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-2">
        <h3 className="font-medium text-neutral-900 text-sm">Results</h3>
        {isLoading && (
          <span className="text-neutral-500 text-xs">Processing...</span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {visibleVerdicts.map((verdict, idx) => {
          const sources = verdict.sources || [];
          const visibleSources = sources.slice(0, MAX_VISIBLE_SOURCES);
          const hiddenSources = sources.slice(MAX_VISIBLE_SOURCES);

          return (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="rounded-lg border border-neutral-200 bg-white p-3"
              initial={{ opacity: 0, y: 5 }}
              key={`${verdict.text.slice(0, 20)}-${idx}`}
              transition={{ duration: 0.2, delay: idx * 0.05 }}
            >
              <div className="mb-2 flex items-start justify-between gap-2">
                <VerdictBadge verdict={verdict} />
                {sources.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1">
                    {visibleSources.map((source) => (
                      <SourceLink key={source.url} source={source} />
                    ))}
                    {hiddenSources.length > 0 && (
                      <AdditionalSourcesPopover sources={hiddenSources} />
                    )}
                  </div>
                )}
              </div>

              <p className="mb-2 font-medium text-neutral-900 text-sm">
                {verdict.text}
              </p>

              {verdict.reasoning && (
                <p className="text-neutral-600 text-xs leading-relaxed">
                  {verdict.reasoning}
                </p>
              )}
            </motion.div>
          );
        })}
      </div>

      {hasMoreCards && (
        <ToggleButton
          onClick={() => setShowAllCards(!showAllCards)}
          remainingCount={remainingCount}
          show={showAllCards}
        />
      )}
    </div>
  );
});
