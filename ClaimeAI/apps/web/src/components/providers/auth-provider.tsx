"use client";

import { ClerkProvider } from "@clerk/nextjs";
import type { Theme } from "@clerk/types";
import type { ComponentProps } from "react";

export const AuthProvider = (
  properties: ComponentProps<typeof ClerkProvider>
) => {
  const elements: Theme["elements"] = {
    dividerLine: "bg-border",
    socialButtonsIconButton: "bg-card",
    navbarButton: "text-foreground",
    organizationSwitcherTrigger__open: "bg-background",
    organizationPreviewMainIdentifier: "text-foreground",
    organizationSwitcherTriggerIcon: "text-muted-foreground",
    organizationPreview__organizationSwitcherTrigger: "gap-2",
    organizationPreviewAvatarContainer: "shrink-0",
  };

  return <ClerkProvider {...properties} appearance={{ elements }} />;
};
