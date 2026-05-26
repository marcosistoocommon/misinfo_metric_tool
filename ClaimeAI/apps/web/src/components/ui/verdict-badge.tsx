import { motion } from "framer-motion";
import {
  AlertCircle,
  Check,
  CircleCheck,
  CircleSlash,
  Info,
  X,
} from "lucide-react";
import { Badge, type BadgeProps } from "@/components/ui/badge";
import type { Claim } from "@/lib/store";
import { cn } from "@/lib/utils";

interface VerdictBadgeProps {
  verdict: Claim;
}

const getBadgeVariant = (result: string | undefined): BadgeProps["variant"] => {
  switch (result) {
    case "Supported":
      return "success-subtle";
    case "Refuted":
      return "destructive-subtle";
    default:
      return "secondary";
  }
};

const getIcon = (result: string | undefined) => {
  switch (result) {
    case "Supported":
      return <CircleCheck className="mt-px mr-1 size-3.5 flex-shrink-0" />;
    case "Refuted":
      return <CircleSlash className="mt-px mr-1 size-3.5 flex-shrink-0" />;
    default:
      return null;
  }
};

export const VerdictBadge = ({ verdict }: VerdictBadgeProps) => {
  if (!verdict.result) return null;

  return (
    <Badge
      className="flex w-fit items-center justify-center rounded-sm py-0.5 pr-2 pl-1 text-[11px]"
      variant={getBadgeVariant(verdict.result)}
    >
      {getIcon(verdict.result)}
      <motion.span
        animate={{ opacity: 1, x: 0 }}
        className="truncate"
        initial={{ opacity: 0, x: -2 }}
        transition={{ duration: 0.2 }}
      >
        {verdict.result}
      </motion.span>
    </Badge>
  );
};
