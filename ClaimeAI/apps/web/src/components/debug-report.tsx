"use client";

import { motion } from "framer-motion";
import { useFactCheckerStore } from "@/lib/store";

export const DebugReport = () => {
  const { rawServerEvents } = useFactCheckerStore();

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
      initial={{ opacity: 0, y: 10 }}
      transition={{ duration: 0.3 }}
    >
      <div className="mt-6">
        <h3 className="mb-3 font-medium text-neutral-900 text-sm">
          Debug Information
        </h3>
        <div className="max-h-96 overflow-y-auto rounded-lg bg-neutral-50 p-3">
          <pre className="text-xs">
            {JSON.stringify(rawServerEvents, null, 2)}
          </pre>
        </div>
      </div>
    </motion.div>
  );
};
