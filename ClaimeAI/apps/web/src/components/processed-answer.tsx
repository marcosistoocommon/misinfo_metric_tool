import { motion } from "framer-motion";
import { Citation } from "@/components/ui/citation";
import type { Claim } from "@/lib/store";
import { cn } from "@/lib/utils";

interface ProcessedAnswerProps {
  claims: Map<string, Claim[]>;
  expandedCitation: number | null;
  setExpandedCitation: (id: number | null) => void;
  hasNoClaims?: boolean;
  isLoading?: boolean;
}

const getVerdictBorderColor = (claims: Claim[]): string => {
  if (!claims?.length) return "";

  const results = claims
    .filter((claim) => claim.status === "verified" && claim.result)
    .map((claim) => claim.result) as string[];

  if (results.length === 0) return "";

  if (results.includes("Refuted")) return "border-l-red-500";
  if (results.includes("Supported")) return "border-l-emerald-500";

  return "";
};

export const ProcessedAnswer = ({
  claims,
  expandedCitation,
  setExpandedCitation,
  hasNoClaims = false,
  isLoading = false,
}: ProcessedAnswerProps) => {
  if (claims.size === 0) {
    if (hasNoClaims) {
      return (
        <div className="py-8 text-center">
          <p className="text-neutral-500 text-sm">No factual claims found to verify.</p>
        </div>
      );
    }
    
    return (
      <div className="py-8 text-center">
        <p className="text-neutral-500 text-sm">
          {isLoading ? "Analyzing content..." : "No claims available."}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-1 text-neutral-900 text-sm leading-relaxed">
      {Array.from(claims.entries()).map(([sentence, sentenceClaims], idx) => {
        const borderColor = getVerdictBorderColor(sentenceClaims);
        const hasClaims = sentenceClaims.length > 0;
        const isExpanded = expandedCitation === idx;

        return (
          <motion.button
            animate={{ opacity: 1, y: 0 }}
            className={cn(
              "inline-block rounded-md px-2 py-1.5 text-start transition-colors duration-200",
              hasClaims
                ? `border bg-white hover:bg-neutral-50 ${borderColor ? `border-l-2 ${borderColor}` : "border-neutral-200"}`
                : "border border-neutral-300 border-dashed text-neutral-600"
            )}
            initial={{ opacity: 0, y: 5 }}
            key={sentence}
            onClick={() => setExpandedCitation(isExpanded ? null : idx)}
            transition={{ duration: 0.2, delay: Math.min(idx * 0.02, 0.5) }}
            type="button"
          >
            {sentence}
            {hasClaims && (
              <Citation
                claim={sentenceClaims}
                id={idx}
                isExpanded={isExpanded}
                onClick={() => setExpandedCitation(isExpanded ? null : idx)}
              />
            )}
          </motion.button>
        );
      })}
    </div>
  );
};
