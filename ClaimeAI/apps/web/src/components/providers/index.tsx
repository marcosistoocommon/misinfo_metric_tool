import { ClerkProvider } from "@clerk/nextjs";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryProvider } from "./query-provider";

export const Providers = ({ children }: Readonly<React.PropsWithChildren>) => (
  <ClerkProvider>
    <NuqsAdapter>
      <QueryProvider>
        <TooltipProvider>{children} </TooltipProvider>
      </QueryProvider>
    </NuqsAdapter>
  </ClerkProvider>
);
