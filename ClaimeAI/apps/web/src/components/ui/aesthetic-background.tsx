"use client";

import { motion } from "framer-motion";
import { memo, useMemo } from "react";
import { cn } from "@/lib/utils";

const phrases = [
  "Extracting factual claims...",
  "Verifying entailment relationships.",
  "Analyzing claim decontextualization.",
  "Assessing factual coverage.",
  "Evaluating claim precision.",
  "Disambiguating reference ambiguity.",
  "Detecting linguistic ambiguity.",
  "Cross-referencing reliable sources.",
  "Computing verifiability metrics.",
  "Decomposing complex statements.",
  "Resolving structural ambiguity.",
  "Maximally clarifying context.",
  "Validating referential integrity.",
  "Verifying propositional entailment.",
  "Fact-checking in progress...",
  "Performing claim extraction.",
  "Decomposing for verification.",
  "Resolving temporal ambiguity.",
  "Evaluating claim atomicity.",
  "Checking source credibility.",
  "Measuring element-level coverage.",
  "Verifying factual precision.",
  "Claimify methodology active.",
  "Implementing entailment checks.",
  "Analyzing sentence elements.",
  "Computing MaxClarifiedSentence.",
  "Contextual sentence analysis.",
  "Applying SAFE methodology.",
  "Identifying check-worthy claims.",
  "Verifying objective information.",
  "Cross-referencing evidence sets.",
  "Applying the Statements and Actions Rule.",
  "Analyzing decomposed propositions.",
  "Evaluating factual evidence.",
  "Implementing verification protocol.",
  "Determining veracity assessment.",
  "Measuring factual accuracy.",
  "FActScore computation in progress.",
  "Evaluating for faithful representation.",
  "Maximizing decontextualization.",
];

interface PhraseItemProps {
  text: string;
  highlight: boolean;
  opacity: string;
  rotation: number;
  delay: number;
}

const PhraseItem = ({
  text,
  highlight,
  opacity,
  rotation,
  delay,
}: PhraseItemProps) => (
  <motion.span
    animate={{ opacity: Number.parseFloat(opacity), y: 0 }}
    className={`select-none px-0.5 text-[9px] leading-tight md:text-[10px] ${
      highlight ? "font-medium text-neutral-600" : "text-neutral-400"
    }`}
    initial={{ opacity: 0, y: 5 }}
    style={{
      transform: rotation ? `rotate(${rotation}deg)` : "none",
    }}
    transition={{
      duration: 0.6,
      delay,
      ease: [0.22, 1, 0.36, 1],
    }}
  >
    {text}
  </motion.span>
);

export const AestheticBackground = memo(function AestheticBackground({
  className,
}: {
  className?: string;
}) {
  const extendedPhrases = useMemo(() => {
    // Reduced from 25 to 15 repetitions for better performance
    const base = new Array(15).fill(null).flatMap(() => phrases);
    return base.map((phrase, index) => ({
      text: phrase,
      highlight: index % 6 === 0,
      opacity: (Math.random() * 0.35 + 0.08).toFixed(2),
      rotation: Math.random() > 0.8 ? Math.random() * 3 - 1.5 : 0,
      delay: Math.random() * 0.5,
    }));
  }, []);

  return (
    <motion.div
      animate={{ opacity: 1 }}
      aria-hidden="true"
      className={cn(
        "pointer-events-none fixed inset-x-0 bottom-0 z-[-1] h-1/2 overflow-hidden",
        className
      )}
      initial={{ opacity: 0 }}
      transition={{ duration: 0.8, delay: 0.2 }}
    >
      <div className="absolute inset-0">
        <motion.div
          animate={{ opacity: 1 }}
          className={cn(
            "absolute inset-0 z-20 bg-gradient-to-t from-white/90 via-white/60 to-transparent",
            className
          )}
          initial={{ opacity: 0 }}
          style={{
            maskImage: "linear-gradient(to top, black 60%, transparent 100%)",
            WebkitMaskImage:
              "linear-gradient(to top, black 60%, transparent 100%)",
          }}
          transition={{ duration: 1.2 }}
        />
        <div className="relative h-full w-full p-3 md:p-5 lg:p-6">
          <div className="grid grid-cols-6 gap-x-0.5 gap-y-0.5 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 xl:grid-cols-14">
            {extendedPhrases.map((item, index) => (
              <PhraseItem
                delay={item.delay}
                highlight={item.highlight}
                key={`${item.text}-${index}`}
                opacity={item.opacity}
                rotation={item.rotation}
                text={item.text}
              />
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
});
