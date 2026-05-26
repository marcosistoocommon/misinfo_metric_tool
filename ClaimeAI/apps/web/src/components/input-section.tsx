"use client";

import NumberFlow from "@number-flow/react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ArrowUpIcon, type ArrowUpIconHandle } from "@/components/ui/icons";
import { useOptimisticCheckCreation } from "@/hooks/use-checks";
import { MAX_INPUT_LIMIT } from "@/lib/constants";
import { useFactCheckerStore } from "@/lib/store";
import { cn, generateCheckId } from "@/lib/utils";

const useInputManagement = () => {
  const { text, setText, isLoading, startVerification } = useFactCheckerStore();
  const { addOptimisticCheck, generateTitle } = useOptimisticCheckCreation();
  const [hasReachedLimit, setHasReachedLimit] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const encodedAnswer = searchParams.get("a");
    if (encodedAnswer) {
      try {
        const decodedAnswer = decodeURIComponent(encodedAnswer);
        setText(decodedAnswer);
      } catch {
        setText("(Error decoding answer)");
      }
    }
  }, [setText]);

  const characterCount = text.length;
  const isOverLimit = characterCount >= MAX_INPUT_LIMIT;
  const isNearLimit = characterCount > MAX_INPUT_LIMIT * 0.8 && !isOverLimit;

  const handleTextChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const inputText = event.target.value;
      if (inputText.length > MAX_INPUT_LIMIT) {
        setText(inputText.slice(0, MAX_INPUT_LIMIT));
        setHasReachedLimit(true);
        setTimeout(() => setHasReachedLimit(false), 500);
      } else {
        setText(inputText);
      }
    },
    [setText]
  );

  const handleSubmission = useCallback(async () => {
    if (!isLoading && text && !isOverLimit) {
      try {
        const checkId = generateCheckId(text);
        addOptimisticCheck(checkId, text);
        await startVerification(text, checkId);
        generateTitle({ content: text, checkId });
        router.push(`/checks/${checkId}?new=true`);
      } catch (error) {
        console.error("Failed to start verification:", error);
      }
    }
  }, [
    isLoading,
    text,
    isOverLimit,
    addOptimisticCheck,
    startVerification,
    generateTitle,
    router,
  ]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        handleSubmission();
      }
    },
    [handleSubmission]
  );

  return {
    answer: text,
    isLoading,
    characterCount,
    isOverLimit,
    isNearLimit,
    hasReachedLimit,
    handleTextChange,
    handleKeyDown,
    handleSubmission,
  };
};

interface CharacterCounterProps {
  count: number;
  hasReachedLimit: boolean;
  isOverLimit: boolean;
  isNearLimit: boolean;
}

const CharacterCounter = ({
  count,
  hasReachedLimit,
  isOverLimit,
  isNearLimit,
}: CharacterCounterProps) => (
  <motion.div
    animate={{
      x: hasReachedLimit ? [-1, 1, -1, 1, 0] : 0,
    }}
    className="flex items-center gap-1.5 pl-2 text-xs"
    transition={{ duration: 0.3, ease: "easeInOut" }}
  >
    <NumberFlow
      className={cn(
        "font-medium tabular-nums transition-colors",
        isOverLimit
          ? "text-red-500"
          : isNearLimit
            ? "text-amber-500"
            : "text-neutral-400"
      )}
      value={count}
    />
    <span className="text-neutral-300">/</span>
    <span className="font-medium text-neutral-400">
      {MAX_INPUT_LIMIT.toLocaleString()}
    </span>
  </motion.div>
);

interface SubmitButtonProps {
  isLoading: boolean;
  isDisabled: boolean;
  onSubmit: () => void;
}

const SubmitButton = ({
  isLoading,
  isDisabled,
  onSubmit,
}: SubmitButtonProps) => {
  const arrowIconRef = useRef<ArrowUpIconHandle>(null);

  const handleMouseEnter = () => arrowIconRef.current?.startAnimation();
  const handleMouseLeave = () => arrowIconRef.current?.stopAnimation();

  return (
    <Button
      className="size-8 border border-transparent bg-neutral-800 transition-all hover:bg-neutral-700 disabled:border-neutral-400/60 disabled:bg-neutral-100 disabled:text-neutral-500"
      disabled={isLoading || isDisabled}
      onClick={onSubmit}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      size="icon"
    >
      {isLoading ? (
        <div className="size-4 animate-spin rounded-full border-2 border-neutral-700 border-t-transparent" />
      ) : (
        <ArrowUpIcon ref={arrowIconRef} />
      )}
    </Button>
  );
};

export const InputSection = () => {
  const {
    answer,
    isLoading,
    characterCount,
    isOverLimit,
    isNearLimit,
    hasReachedLimit,
    handleTextChange,
    handleKeyDown,
    handleSubmission,
  } = useInputManagement();

  return (
    <motion.section
      animate={{ opacity: 1 }}
      className="relative max-h-32 min-h-32 w-full rounded-xl border bg-white p-2"
      initial={{ opacity: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      <textarea
        className={cn(
          "w-full max-w-5xl resize-none rounded-sm border-neutral-200 bg-white pr-20 pb-8 text-sm transition-colors placeholder:text-neutral-400",
          "focus:outline-none focus:ring-0",
          isOverLimit && "border-red-200 focus:border-red-300"
        )}
        disabled={isLoading}
        onChange={handleTextChange}
        onKeyDown={handleKeyDown}
        placeholder="Drop any claim here to see if it stands up to the facts..."
        value={answer}
      />
      <div className="flex items-center justify-between bg-white">
        <CharacterCounter
          count={characterCount}
          hasReachedLimit={hasReachedLimit}
          isNearLimit={isNearLimit}
          isOverLimit={isOverLimit}
        />
        <SubmitButton
          isDisabled={!answer || isOverLimit}
          isLoading={isLoading}
          onSubmit={handleSubmission}
        />
      </div>
    </motion.section>
  );
};
