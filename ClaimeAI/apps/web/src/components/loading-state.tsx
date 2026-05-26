import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface LoadingStateProps {
  message: string;
}

export const LoadingState = ({ message }: LoadingStateProps) => (
  <div className="flex flex-col items-center justify-center py-24">
    <motion.div
      animate={{ rotate: 360 }}
      className={cn(
        "mb-2.5 size-6 rounded-full border-2 border-blue-100 border-t-blue-600"
      )}
      transition={{
        duration: 1,
        repeat: Number.POSITIVE_INFINITY,
        ease: "linear",
      }}
    />
    <motion.p
      animate={{ opacity: 1, y: 0 }}
      className="text-neutral-500 text-xs"
      initial={{ opacity: 0, y: 5 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      {message}
    </motion.p>
  </div>
);

export const ProcessingIndicator = ({ message }: { message: string }) => (
  <motion.div
    animate={{ opacity: 1, y: 0 }}
    className={cn(
      "flex items-center gap-2.5 rounded-[4px] border border-blue-100 bg-blue-50 px-3 py-2"
    )}
    initial={{ opacity: 0, y: -5 }}
    transition={{ duration: 0.2 }}
  >
    <motion.div
      animate={{ rotate: 360 }}
      className="size-3 rounded-full border-[1.5px] border-blue-100 border-t-blue-500"
      transition={{
        duration: 1.2,
        repeat: Number.POSITIVE_INFINITY,
        ease: "linear",
      }}
    />
    <p className="font-medium text-blue-600 text-xs">{message}</p>
  </motion.div>
);
