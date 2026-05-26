"use client";

import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { Claim } from "@/lib/store";

import { LoadingState } from "./loading-state";
import { ProcessedAnswer } from "./processed-answer";
import { SourcePills } from "./source-pills";
import { VerdictProgress } from "./verdict-progress";
import { VerdictSummary } from "./verdict-summary";

interface FactCheckerProps {
  claims: Map<string, Claim[]>;
  isLoading: boolean;
  hasNoClaims?: boolean;
}

export const FactChecker = ({
  claims,
  isLoading,
  hasNoClaims,
}: FactCheckerProps) => {
  const [expandedCitation, setExpandedCitation] = useState<number | null>(null);

  const { verifiedClaims, hasSentences, hasAnyClaims, hasVerifiedClaims } =
    useMemo(() => {
      const allClaims = Array.from(claims.values()).flat();
      const verifiedClaims = allClaims.filter(
        (claim) => claim.status === "verified"
      );

      return {
        allClaims,
        verifiedClaims,
        hasSentences: claims.size > 0,
        hasAnyClaims: allClaims.length > 0,
        hasVerifiedClaims: verifiedClaims.length > 0,
      };
    }, [claims]);

  const showInitialLoading = isLoading && !hasSentences;
  const showProcessingIndicator =
    isLoading && hasSentences && !(hasAnyClaims && hasVerifiedClaims);

  if (showInitialLoading) {
    return <LoadingState message="Analyzing content..." />;
  }

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="w-full space-y-6"
      initial={{ opacity: 0, y: 10 }}
      transition={{ duration: 0.3 }}
    >
      {hasVerifiedClaims && (
        <motion.div
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
          initial={{ opacity: 0, y: 5 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <SourcePills verdicts={verifiedClaims} />
          <VerdictProgress isLoading={isLoading} verdicts={verifiedClaims} />
        </motion.div>
      )}

      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-neutral-900 text-sm">Claims</h3>
          {showProcessingIndicator && (
            <div className="flex items-center gap-2">
              <div className="h-1 w-1 animate-pulse rounded-full bg-blue-500" />
              <div
                className="h-1 w-1 animate-pulse rounded-full bg-blue-500"
                style={{ animationDelay: "0.2s" }}
              />
              <div
                className="h-1 w-1 animate-pulse rounded-full bg-blue-500"
                style={{ animationDelay: "0.4s" }}
              />
              <span className="text-neutral-500 text-xs">
                {hasAnyClaims ? "Verifying claims..." : "Extracting claims..."}
              </span>
            </div>
          )}
        </div>
        <ProcessedAnswer
          claims={claims}
          expandedCitation={expandedCitation}
          setExpandedCitation={setExpandedCitation}
          hasNoClaims={hasNoClaims}
          isLoading={isLoading}
        />
      </div>

      {hasVerifiedClaims && (
        <VerdictSummary isLoading={isLoading} verdicts={verifiedClaims} />
      )}
    </motion.div>
  );
};
