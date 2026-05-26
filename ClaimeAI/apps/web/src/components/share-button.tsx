"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check, Copy, Globe, Lock, Share } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { client } from "@/lib/rpc";

interface ShareButtonProps {
  checkId: string;
  disabled?: boolean;
  initialIsPublic: boolean;
}

export const ShareButton = ({
  checkId,
  disabled,
  initialIsPublic,
}: ShareButtonProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isPublic, setIsPublic] = useState(initialIsPublic);
  const [copied, setCopied] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const handleToggleVisibility = async (makePublic: boolean) => {
    if (!checkId) return;
    setIsLoading(true);

    try {
      const response = await client.api.agent.checks[":checkId"].public.$patch({
        param: { checkId },
        json: { isPublic: makePublic },
      });

      if (response.ok) {
        const data = await response.json();
        setIsPublic(data.isPublic);
      }
    } catch (error) {
      console.error("Visibility toggle error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async (url: string) => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Copy error:", error);
    }
  };

  return (
    <Popover onOpenChange={setIsOpen} open={isOpen}>
      <PopoverTrigger asChild>
        <Button
          className="size-9 border-zinc-200 bg-white transition-colors duration-150 hover:bg-zinc-50"
          disabled={disabled}
          size="icon"
          variant="outline"
        >
          <Share className="size-3.5 text-zinc-600" />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        align="end"
        asChild
        className="w-72 rounded-lg border-zinc-200 bg-white p-0 shadow-sm"
        side="bottom"
      >
        <motion.div
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -4 }}
          initial={{ opacity: 0, scale: 0.95, y: -4 }}
          transition={{ duration: 0.15, ease: [0.23, 1, 0.32, 1] }}
        >
          <div className="space-y-3 p-4">
            <div>
              <h4 className="font-medium text-sm text-zinc-900">Share</h4>
              <p className="mt-0.5 text-xs text-zinc-500">
                {isPublic ? "Public link ready" : "Make public to share"}
              </p>
            </div>

            <AnimatePresence mode="wait">
              {isPublic ? (
                <motion.div
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-2.5"
                  exit={{ opacity: 0, y: -4 }}
                  initial={{ opacity: 0, y: 4 }}
                  key="share"
                  transition={{ duration: 0.12 }}
                >
                  <div className="flex items-center gap-1.5">
                    <div className="flex-1 truncate rounded border border-zinc-200 bg-zinc-50 px-2.5 py-1.5 font-mono text-xs text-zinc-600">
                      {window.location.origin}/public/{checkId}
                    </div>
                    <Button
                      className="size-7.5 rounded border-zinc-200 p-0 transition-colors duration-150 hover:bg-zinc-50"
                      onClick={() =>
                        handleCopy(
                          `${window.location.origin}/public/${checkId}`
                        )
                      }
                      size="sm"
                      variant="outline"
                    >
                      <AnimatePresence mode="wait">
                        {copied ? (
                          <motion.div
                            animate={{ scale: 1 }}
                            exit={{ scale: 0 }}
                            initial={{ scale: 0 }}
                            key="check"
                            transition={{ duration: 0.1 }}
                          >
                            <Check className="size-3 text-zinc-600" />
                          </motion.div>
                        ) : (
                          <motion.div
                            animate={{ scale: 1 }}
                            exit={{ scale: 0 }}
                            initial={{ scale: 0 }}
                            key="copy"
                            transition={{ duration: 0.1 }}
                          >
                            <Copy className="size-3 text-zinc-500" />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </Button>
                  </div>

                  <Button
                    className="h-7 w-full border-zinc-200 text-xs text-zinc-600 transition-colors duration-150 hover:bg-zinc-50"
                    disabled={isLoading}
                    onClick={() => handleToggleVisibility(false)}
                    variant="outline"
                  >
                    <Lock className="mr-1.5 size-3" />
                    {isLoading ? "Making private..." : "Make private"}
                  </Button>
                </motion.div>
              ) : (
                <motion.div
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  initial={{ opacity: 0, y: 4 }}
                  key="publish"
                  transition={{ duration: 0.12 }}
                >
                  <Button
                    className="h-8 w-full bg-zinc-900 text-sm text-white transition-colors duration-150 hover:bg-zinc-800"
                    disabled={isLoading}
                    onClick={() => handleToggleVisibility(true)}
                  >
                    <Globe className="mr-1.5 size-3.5" />
                    {isLoading ? "Publishing..." : "Publish"}
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </PopoverContent>
    </Popover>
  );
};
