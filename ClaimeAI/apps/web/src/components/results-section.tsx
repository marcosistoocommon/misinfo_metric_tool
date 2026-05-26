"use client";

import { motion } from "framer-motion";
import { useFactCheckerStore } from "@/lib/store";
import { FactChecker } from "./fact-checker";

export const ResultsSection = () => {
  const { isLoading, claims, hasNoClaims } = useFactCheckerStore();

  return (
    <motion.section
      animate={{ opacity: 1 }}
      aria-busy={isLoading}
      aria-label="Fact check results"
      aria-live="polite"
      className="flex w-full flex-grow flex-col lg:row-span-2"
      initial={{ opacity: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      {(claims.size > 0 || hasNoClaims) && (
        <article className="flex w-full flex-grow flex-col">
          <h2 className="sr-only">Fact Check Results</h2>
          <FactChecker
            claims={claims}
            isLoading={isLoading}
            hasNoClaims={hasNoClaims}
          />
        </article>
      )}
    </motion.section>
  );
};
