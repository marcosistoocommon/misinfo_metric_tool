import { AnimatePresence, motion } from "framer-motion";
import { ClipboardCheck, Quote, Scale } from "lucide-react";
import type React from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import type { Claim } from "@/lib/store";
import { cn } from "@/lib/utils";
import { VerdictBadge } from "./verdict-badge";

interface CitationProps {
  id: number;
  claim: Claim[];
  isExpanded: boolean;
  onClick: () => void;
}

interface CitationSectionProps {
  title: string;
  items: Claim[];
  className?: string;
  delay: number;
  icon: React.ComponentType<{ className?: string }>;
  renderItem: (item: Claim, idx: number) => React.ReactNode;
}

const CitationSection = ({
  title,
  items,
  className,
  delay,
  icon: Icon,
  renderItem,
}: CitationSectionProps) => {
  if (items.length === 0) return null;

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      initial={{ opacity: 0, y: 5 }}
      transition={{ duration: 0.2, delay }}
    >
      <div className="mb-2 flex items-center gap-1.5">
        <Icon className="h-3.5 w-3.5 text-neutral-500" />
        <h5 className="font-medium text-neutral-600 text-sm">{title}</h5>
      </div>
      <div
        className={cn(
          title === "Extracted Claims" &&
            "rounded-md border border-neutral-200 bg-neutral-50",
          className
        )}
      >
        {items.map(renderItem)}
      </div>
    </motion.div>
  );
};

export const Citation = ({ id, claim, isExpanded, onClick }: CitationProps) => (
  <Sheet
    onOpenChange={(open) => {
      if (!open) onClick();
    }}
    open={isExpanded}
  >
    <SheetTrigger asChild>
      <motion.span
        className="mb-1 ml-1 cursor-pointer text-mono text-neutral-800 text-xs transition-colors hover:text-neutral-600"
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        [{id + 1}]
      </motion.span>
    </SheetTrigger>
    <SheetContent
      className="my-auto w-full min-w-md max-w-md gap-0 rounded-r-2xl shadow-lg"
      onClick={(e) => e.stopPropagation()}
      side="right"
    >
      <SheetHeader className="border-b pb-3">
        <SheetTitle className="flex items-center gap-2 text-left">
          <Quote className="h-4 w-4 text-neutral-600" />
          Claim Details{" "}
          <span className="text-neutral-500 text-xs">#{id + 1}</span>
        </SheetTitle>
      </SheetHeader>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            animate={{ opacity: 1, x: 0 }}
            className="flex-1 overflow-y-auto"
            exit={{ opacity: 0, x: 20 }}
            initial={{ opacity: 0, x: 20 }}
            onClick={(e) => e.stopPropagation()}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="space-y-4 p-4">
              <CitationSection
                delay={0.1}
                icon={ClipboardCheck}
                items={claim}
                renderItem={(item, idx) => (
                  <div
                    className="border-neutral-200 border-b p-3 text-neutral-900 text-sm last:border-b-0"
                    key={idx}
                  >
                    {item.text}
                  </div>
                )}
                title="Extracted Claims"
              />

              <CitationSection
                className="space-y-2"
                delay={0.2}
                icon={Scale}
                items={claim.filter((c) => c.status === "verified")}
                renderItem={(verdict, idx) => (
                  <div
                    className="overflow-hidden rounded-lg border border-neutral-200 bg-white"
                    key={idx}
                  >
                    <div className="p-4">
                      <div className="mb-3">
                        <VerdictBadge verdict={verdict} />
                      </div>
                      <p className="mb-3 font-medium text-neutral-900 text-sm">
                        {verdict.text}
                      </p>
                      {verdict.reasoning && (
                        <p className="text-neutral-600 text-sm leading-relaxed">
                          {verdict.reasoning}
                        </p>
                      )}
                      {verdict.sources && verdict.sources.length > 0 && (
                        <div className="mt-3 border-neutral-100 border-t pt-3">
                          <div className="mb-2 font-medium text-neutral-600 text-sm">
                            Sources:
                          </div>
                          <div className="space-y-2">
                            {verdict.sources.map((source, sidx) => (
                              <a
                                className="block text-blue-600 text-sm hover:underline"
                                href={source.url}
                                key={`${source.url}-${sidx}`}
                                rel="noopener noreferrer"
                                target="_blank"
                              >
                                {source.title || source.url}
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                title="Verdicts"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </SheetContent>
  </Sheet>
);
