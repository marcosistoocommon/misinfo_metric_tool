import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface StreamingIndicatorProps {
  isActive: boolean;
}

export const StreamingIndicator = ({ isActive }: StreamingIndicatorProps) => {
  if (!isActive) return null;
  return (
    <motion.div
      animate={{ opacity: 1, x: 0 }}
      className={cn("ml-2 flex items-center gap-1 font-medium text-blue-500")}
      exit={{ opacity: 0, x: 10 }}
      initial={{ opacity: 0, x: -10 }}
      transition={{ duration: 0.2 }}
    >
      <div className="relative flex gap-0.5">
        <motion.div
          animate={{ scale: [1, 1.5, 1] }}
          className="h-[3px] w-[3px] rounded-full bg-blue-500"
          transition={{
            duration: 1.2,
            repeat: Number.POSITIVE_INFINITY,
            delay: 0,
          }}
        />
        <motion.div
          animate={{ scale: [1, 1.5, 1] }}
          className="h-[3px] w-[3px] rounded-full bg-blue-500"
          transition={{
            duration: 1.2,
            repeat: Number.POSITIVE_INFINITY,
            delay: 0.4,
          }}
        />
        <motion.div
          animate={{ scale: [1, 1.5, 1] }}
          className="h-[3px] w-[3px] rounded-full bg-blue-500"
          transition={{
            duration: 1.2,
            repeat: Number.POSITIVE_INFINITY,
            delay: 0.8,
          }}
        />
      </div>
      <span className="text-xs">streaming</span>
    </motion.div>
  );
};
