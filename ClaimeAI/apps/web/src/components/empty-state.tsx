"use client";

import { motion } from "framer-motion";
import { Bot, FileText, Search, Shield } from "lucide-react";
import { FeatureCard } from "@/components/feature-card";

const FEATURES = [
  {
    icon: Bot,
    title: "AI Analysis",
    description: "Extract and verify claims",
  },
  {
    icon: Search,
    title: "Evidence Search",
    description: "Real-time web validation",
  },
  {
    icon: Shield,
    title: "Verification",
    description: "Multi-source evaluation",
  },
  {
    icon: FileText,
    title: "Reports",
    description: "Detailed citations",
  },
] as const;

export const EmptyState = () => (
  <motion.div
    animate={{ opacity: 1 }}
    className="flex w-full flex-col items-center justify-center space-y-8 text-center"
    initial={{ opacity: 0 }}
    transition={{ duration: 0.4 }}
  >
    <div className="space-y-4">
      <h1 className="font-semibold text-3xl text-neutral-900 tracking-tight">
        Text In. Truth Out.
      </h1>
    </div>

    <div className="flex w-full gap-3 px-2">
      {FEATURES.map((feature) => (
        <FeatureCard key={feature.title} {...feature} />
      ))}
    </div>
  </motion.div>
);
