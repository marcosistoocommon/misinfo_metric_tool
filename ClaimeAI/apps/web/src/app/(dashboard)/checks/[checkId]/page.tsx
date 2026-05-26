"use client";

import { useParams } from "next/navigation";
import { useEffect, useRef } from "react";
import { CheckHeader } from "@/components/check-header";
import { FactChecker } from "@/components/fact-checker";
import { useFactCheckerStore } from "@/lib/store";

const CheckPage = () => {
  const { checkId } = useParams();
  const connectToStream = useFactCheckerStore((state) => state.connectToStream);
  const connectionInitiated = useRef(false);
  const { isLoading, claims, hasNoClaims } = useFactCheckerStore();
  const { metadata } = useFactCheckerStore();

  useEffect(() => {
    if (checkId && !connectionInitiated.current) {
      connectionInitiated.current = true;
      connectToStream(checkId as string);
    }
  }, [checkId, connectToStream]);

  return (
    <div className="container mx-auto max-w-6xl space-y-8 p-8">
      <CheckHeader
        checkId={checkId as string}
        claims={claims}
        isLoading={isLoading}
        metadata={metadata}
      />
      <FactChecker
        claims={claims}
        isLoading={isLoading}
        hasNoClaims={hasNoClaims}
      />
    </div>
  );
};

export default CheckPage;
