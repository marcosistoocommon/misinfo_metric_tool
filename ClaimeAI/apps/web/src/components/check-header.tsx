import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import { useEffect, useState } from "react";
import { ShareButton } from "@/components/share-button";
import type { Claim } from "@/lib/store";
import type { CheckMetadata } from "@/types";

interface CheckHeaderProps {
  metadata?: CheckMetadata | null;
  claims?: Map<string, Claim[]>;
  checkId: string;
  isLoading?: boolean;
}

const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  }).format(date);
};

export const CheckHeader = ({
  metadata,
  claims,
  checkId,
  isLoading = false,
}: CheckHeaderProps) => {
  const [showSkeleton, setShowSkeleton] = useState(true);

  useEffect(() => {
    // Show skeleton for 1 second initially
    const timer = setTimeout(() => {
      setShowSkeleton(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const shouldShowTitle = metadata?.title;
  const shouldShowSkeleton = showSkeleton && !shouldShowTitle;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="font-bold text-3xl text-neutral-900 tracking-tight">
          <AnimatePresence mode="wait">
            {shouldShowSkeleton ? (
              <motion.div
                animate={{ opacity: 1 }}
                className="h-9 w-64 animate-pulse rounded bg-neutral-200"
                exit={{ opacity: 0 }}
                initial={{ opacity: 1 }}
                key="skeleton"
                transition={{ duration: 0.3, ease: "easeOut" }}
              />
            ) : shouldShowTitle ? (
              <motion.span
                animate={{ opacity: 1, y: 0 }}
                initial={{ opacity: 0, y: 10 }}
                key="title"
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                {metadata.title}
              </motion.span>
            ) : (
              <motion.div
                animate={{ opacity: 1, y: 0 }}
                initial={{ opacity: 0, y: 10 }}
                key="fallback"
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                Claime Results
              </motion.div>
            )}
          </AnimatePresence>
        </h1>
      </div>

      <p className="text-neutral-700 leading-relaxed">
        {metadata?.text ?? (
          <div className="space-y-2">
            <div className="h-4 animate-pulse rounded bg-neutral-200" />
            <div className="h-4 w-4/5 animate-pulse rounded bg-neutral-200" />
            <div className="h-4 w-3/5 animate-pulse rounded bg-neutral-200" />
          </div>
        )}
      </p>

      <div className="flex items-center justify-between border-neutral-200">
        <div className="flex items-center gap-3">
          {metadata?.user ? (
            <div className="flex items-center gap-2">
              <div className="relative">
                {metadata.user.image ? (
                  <Image
                    alt={metadata.user.email}
                    className="rounded-full border ring-2 ring-neutral-200"
                    height={30}
                    src={metadata.user.image}
                    width={30}
                  />
                ) : (
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-600 font-medium text-sm text-white">
                    {metadata.user.email.charAt(0).toUpperCase()}
                  </div>
                )}
              </div>
              <div>
                <p className="font-medium text-neutral-900 text-sm">
                  {metadata.user.email}
                </p>
                <p className="text-neutral-500 text-xs">
                  {formatDate(metadata.createdAt)}
                </p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 animate-pulse rounded-full bg-neutral-200" />
              <div className="space-y-1">
                <div className="h-4 w-24 animate-pulse rounded bg-neutral-200" />
                <div className="h-3 w-20 animate-pulse rounded bg-neutral-200" />
              </div>
            </div>
          )}
        </div>

        {claims && metadata && (
          <ShareButton
            checkId={checkId}
            disabled={!claims || claims.size === 0}
            initialIsPublic={metadata.isPublic}
          />
        )}
      </div>
    </div>
  );
};
