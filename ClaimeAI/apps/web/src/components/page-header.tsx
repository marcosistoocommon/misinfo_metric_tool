"use client";

import { motion } from "framer-motion";
import Image from "next/image";

export const PageHeader = () => (
  <motion.header
    animate={{ opacity: 1, y: 0 }}
    className="z-20 flex h-[6vh] items-center gap-1! py-6"
    initial={{ opacity: 0, y: -10 }}
    role="banner"
    transition={{ duration: 0.3 }}
  >
    <Image
      alt="ClaimeAI"
      className="size-8"
      height={32}
      src="/logo.svg"
      width={32}
    />

    <h1 className="font-semibold text-neutral-900 text-xl tracking-tight">
      ClaimeAI
    </h1>
  </motion.header>
);
