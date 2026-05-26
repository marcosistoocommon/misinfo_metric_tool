"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

export interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
}

export const FeatureCard = ({
  icon: Icon,
  title,
  description,
}: FeatureCardProps) => (
  <motion.div
    animate={{ opacity: 1, y: 0 }}
    className="group flex w-full items-center gap-3 rounded-md border border-neutral-200 bg-white/60 p-3 backdrop-blur-sm transition-all duration-200 hover:bg-white/80"
    initial={{ opacity: 0, y: 5 }}
  >
    <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-sm border-[0.5px] bg-neutral-100/80">
      <Icon className="h-4 w-4 text-neutral-700" strokeWidth={1.5} />
    </div>
    <div className="flex min-w-0 flex-col items-start">
      <h3 className="font-medium text-neutral-900 text-sm leading-tight">
        {title}
      </h3>
      <p className="mt-0.5 text-neutral-600 text-xs leading-tight">
        {description}
      </p>
    </div>
  </motion.div>
);
