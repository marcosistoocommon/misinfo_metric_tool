"use client";

import { motion } from "framer-motion";
import { ExternalLink, Github } from "lucide-react";
import Image from "next/image";

export const PageFooter = () => {
  return (
    <motion.footer
      animate={{ opacity: 1 }}
      className="mt-auto w-full border-t bg-white p-4"
      initial={{ opacity: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="mx-auto max-w-6xl">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          <motion.section
            animate={{ opacity: 1, y: 0 }}
            aria-labelledby="about-heading"
            className="space-y-3"
            initial={{ opacity: 0, y: 5 }}
            transition={{
              duration: 0.4,
              delay: 0.3,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            <h3
              className="font-semibold text-neutral-900 text-sm"
              id="about-heading"
            >
              About ClaimeAI
            </h3>
            <p className="text-neutral-500 text-xs leading-relaxed">
              Our modular LangGraph system breaks down text into individual
              claims, verifies each against reliable sources, and provides a
              detailed report on what's accurate and what's not. Built with
              advanced Claimify methodology for high-precision fact extraction.
            </p>
          </motion.section>

          <motion.section
            animate={{ opacity: 1, y: 0 }}
            aria-labelledby="resources-heading"
            className="space-y-3"
            initial={{ opacity: 0, y: 5 }}
            transition={{
              duration: 0.4,
              delay: 0.4,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            <h3
              className="font-semibold text-neutral-900 text-sm"
              id="resources-heading"
            >
              Resources
            </h3>
            <nav aria-label="Resource links">
              <ul className="space-y-2">
                <li>
                  <a
                    aria-label="View documentation (opens in same tab)"
                    className="flex items-center gap-1 rounded text-neutral-500 text-xs transition-colors hover:text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    href="/#"
                  >
                    Documentation{" "}
                    <ExternalLink aria-hidden="true" className="h-3 w-3" />
                  </a>
                </li>
                <li>
                  <a
                    aria-label="View API reference (opens in same tab)"
                    className="flex items-center gap-1 rounded text-neutral-500 text-xs transition-colors hover:text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    href="/#"
                  >
                    API Reference{" "}
                    <ExternalLink aria-hidden="true" className="h-3 w-3" />
                  </a>
                </li>
                <li>
                  <a
                    aria-label="View privacy policy (opens in same tab)"
                    className="flex items-center gap-1 rounded text-neutral-500 text-xs transition-colors hover:text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    href="/#"
                  >
                    Privacy Policy{" "}
                    <ExternalLink aria-hidden="true" className="h-3 w-3" />
                  </a>
                </li>
              </ul>
            </nav>
          </motion.section>

          <motion.section
            animate={{ opacity: 1, y: 0 }}
            aria-labelledby="research-heading"
            className="space-y-3"
            initial={{ opacity: 0, y: 5 }}
            transition={{
              duration: 0.4,
              delay: 0.5,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            <h3
              className="font-semibold text-neutral-900 text-sm"
              id="research-heading"
            >
              Research
            </h3>
            <nav aria-label="Research papers">
              <ul className="space-y-2">
                <li>
                  <a
                    aria-label="Read Claimify Methodology paper by Metropolitansky & Larson, 2025 (opens in new tab)"
                    className="flex items-center gap-1 rounded text-neutral-500 text-xs transition-colors hover:text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    href="https://arxiv.org/abs/2502.10855"
                    rel="noopener noreferrer"
                    target="_blank"
                  >
                    Claimify Methodology{" "}
                    <ExternalLink aria-hidden="true" className="h-3 w-3" />
                  </a>
                  <div className="text-neutral-400 text-xs">
                    Metropolitansky & Larson, 2025
                  </div>
                </li>
                <li>
                  <a
                    aria-label="Read SAFE Methodology paper by Wei et al., 2024 (opens in new tab)"
                    className="flex items-center gap-1 rounded text-neutral-500 text-xs transition-colors hover:text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    href="https://arxiv.org/abs/2403.18802"
                    rel="noopener noreferrer"
                    target="_blank"
                  >
                    SAFE Methodology{" "}
                    <ExternalLink aria-hidden="true" className="h-3 w-3" />
                  </a>
                  <div className="text-neutral-400 text-xs">
                    Wei et al., 2024
                  </div>
                </li>
              </ul>
            </nav>
          </motion.section>

          <motion.section
            animate={{ opacity: 1, y: 0 }}
            aria-labelledby="connect-heading"
            className="space-y-3"
            initial={{ opacity: 0, y: 5 }}
            transition={{
              duration: 0.4,
              delay: 0.6,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            <h3
              className="font-semibold text-neutral-900 text-sm"
              id="connect-heading"
            >
              Connect
            </h3>
            <a
              aria-label="View ClaimeAI project on GitHub (opens in new tab)"
              className="inline-flex items-center gap-2 rounded-md border border-neutral-200 px-3 py-1.5 text-neutral-700 text-xs transition-colors hover:border-neutral-300 hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              href="https://github.com/BharathXD/claime"
              rel="noopener noreferrer"
              target="_blank"
            >
              <Github aria-hidden="true" className="h-3.5 w-3.5" />
              <span>View on GitHub</span>
            </a>
          </motion.section>
        </div>

        <motion.div
          animate={{ opacity: 1 }}
          className="mt-4 flex flex-col items-center justify-between border-neutral-100 border-t pt-3 sm:flex-row"
          initial={{ opacity: 0 }}
          transition={{ duration: 0.4, delay: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="flex items-center gap-0.5">
            <Image alt="ClaimeAI" height={20} src="/logo.svg" width={20} />
            <p className="font-semibold text-neutral-600 text-sm">ClaimeAI</p>
          </div>
          <p className="mt-3 text-neutral-400 text-xs sm:mt-0">
            © {new Date().getFullYear()} ClaimeAI. All rights reserved.
          </p>
        </motion.div>
      </div>
    </motion.footer>
  );
};
