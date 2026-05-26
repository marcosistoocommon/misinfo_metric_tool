import { motion } from "framer-motion";
import { InfoIcon } from "lucide-react";
import { useMemo } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Claim } from "@/lib/store";
import { cn } from "@/lib/utils";

interface VerdictProgressProps {
  verdicts: Claim[];
  isLoading: boolean;
}

const VERDICT_CONFIG = {
  Supported: {
    bgClass: "bg-emerald-500",
    label: "Supported",
    description: "Claims verified as correct",
  },
  Refuted: {
    bgClass: "bg-red-500",
    label: "Refuted",
    description: "Claims verified as incorrect",
  },
} as const;

export const VerdictProgress = ({
  verdicts,
  isLoading,
}: VerdictProgressProps) => {
  const stats = useMemo(() => {
    const counts = { Supported: 0, Refuted: 0 };

    for (const verdict of verdicts) {
      if (verdict.result === "Supported" || verdict.result === "Refuted") {
        counts[verdict.result]++;
      }
    }

    const total = verdicts.length;
    const percentages = {
      Supported: total ? (counts.Supported / total) * 100 : 0,
      Refuted: total ? (counts.Refuted / total) * 100 : 0,
    };

    return { counts, total, percentages };
  }, [verdicts]);

  if (stats.total === 0) return null;

  return (
    <div className="w-full space-y-2.5">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-neutral-900 text-sm">Analysis</h3>
        <div className="flex items-center gap-1.5">
          <span className="text-neutral-500 text-xs">
            {stats.total} claim{stats.total !== 1 ? "s" : ""} verified
          </span>
          <Tooltip>
            <TooltipTrigger asChild>
              <InfoIcon className="h-3.5 w-3.5 cursor-help text-neutral-400 transition-colors hover:text-neutral-500" />
            </TooltipTrigger>
            <TooltipContent
              className="border-0 bg-black text-white text-xs"
              side="top"
            >
              Distribution of claim analysis results
            </TooltipContent>
          </Tooltip>
        </div>
      </div>

      <div className="relative h-2 w-full overflow-hidden rounded-full bg-neutral-100">
        <motion.div
          animate={{ width: "100%" }}
          className="absolute inset-0 flex"
          initial={{ width: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          {stats.percentages.Supported > 0 && (
            <div
              className="h-full bg-emerald-500"
              style={{ width: `${stats.percentages.Supported}%` }}
            />
          )}
          {stats.percentages.Refuted > 0 && (
            <div
              className="h-full bg-red-500"
              style={{ width: `${stats.percentages.Refuted}%` }}
            />
          )}
        </motion.div>

        {isLoading && (
          <motion.div
            animate={{ x: ["-100%", "200%"] }}
            className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-400/20 to-transparent"
            transition={{
              duration: 1.2,
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
          />
        )}
      </div>

      <div className="flex items-center justify-between">
        {(
          Object.entries(VERDICT_CONFIG) as [
            keyof typeof VERDICT_CONFIG,
            (typeof VERDICT_CONFIG)[keyof typeof VERDICT_CONFIG],
          ][]
        ).map(([type, config]) => {
          const count = stats.counts[type];
          const percentage = Math.round(stats.percentages[type]);

          return (
            <Tooltip key={type}>
              <TooltipTrigger asChild>
                <div className="flex cursor-default items-center gap-1.5">
                  <div
                    className={cn(
                      "size-2 rounded-full",
                      count > 0 ? config.bgClass : "bg-neutral-300"
                    )}
                  />
                  <div
                    className={cn(
                      "text-xs",
                      count === 0 ? "text-neutral-400" : "text-neutral-600"
                    )}
                  >
                    <span className="font-medium">{config.label}</span>
                    <span className="ml-1 tabular-nums">{percentage}%</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent
                className="border-0 bg-black text-white text-xs"
                side="bottom"
              >
                {config.description} · {count} claims
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
};
