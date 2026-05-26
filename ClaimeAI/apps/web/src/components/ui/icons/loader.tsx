import { cn } from "@/lib/utils";

interface LoaderIconProps {
  className?: string;
}

export const LoaderIcon = ({ className }: LoaderIconProps) => (
  <div
    className={cn(
      "size-4 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-600",
      className
    )}
  />
);
