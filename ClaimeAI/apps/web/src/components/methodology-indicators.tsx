"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const METHODOLOGIES = [
  { color: "bg-blue-500", label: "Claimify Methodology (Microsoft Research)" },
  { color: "bg-green-500", label: "SAFE Framework (Google DeepMind)" },
  { color: "bg-purple-500", label: "LangGraph Multi-Agent Orchestration" },
] as const;

export const MethodologyIndicators = () => (
  <motion.div
    animate={{ opacity: 1, y: 0 }}
    className="flex flex-wrap items-center justify-center gap-6 text-neutral-500 text-xs"
    initial={{ opacity: 0, y: 5 }}
    transition={{ delay: 0.2 }}
  >
    {METHODOLOGIES.map(({ color, label }) => (
      <div className="flex items-center gap-2" key={label}>
        <div className={cn("size-2 rounded-full", color)} />
        <span>{label}</span>
      </div>
    ))}
  </motion.div>
);
