"use client";

import { FactChecker } from "@/components/fact-checker";
import { PageFooter } from "@/components/page-footer";
import { PageHeader } from "@/components/page-header";
import { AestheticBackground } from "@/components/ui/aesthetic-background";
import { client } from "@/lib/rpc";
import type { Claim } from "@/lib/store";
import type { InferResponseType } from "hono";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

const getRelativeTime = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return diffInSeconds <= 0 ? "just now" : `${diffInSeconds} secs ago`;
  }

  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return `${diffInMinutes} min${diffInMinutes === 1 ? "" : "s"} ago`;
  }

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return `${diffInHours} hour${diffInHours === 1 ? "" : "s"} ago`;
  }

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 30) {
    return `${diffInDays} day${diffInDays === 1 ? "" : "s"} ago`;
  }

  const diffInMonths = Math.floor(diffInDays / 30);
  if (diffInMonths < 12) {
    return `${diffInMonths} month${diffInMonths === 1 ? "" : "s"} ago`;
  }

  const diffInYears = Math.floor(diffInMonths / 12);
  return `${diffInYears} year${diffInYears === 1 ? "" : "s"} ago`;
};

const PublicCheckPage = () => {
  const { checkId } = useParams();
  const [claims, setClaims] = useState<Map<string, Claim[]>>(new Map());
  const [checkData, setCheckData] = useState<{
    title: string | null;
    textContent: string | null;
    completedAt: string;
    user: {
      id: string;
      email: string;
      image: string | null;
    } | null;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const extractClaims = (
      resultEvents: InferResponseType<
        (typeof client.api.agent.public)[":checkId"]["$get"],
        200
      >
    ): Map<string, Claim[]> => {
      const claimsMap = new Map<string, Claim[]>();

      for (const item of resultEvents.check) {
        const [sentence, claims] = item;
        claimsMap.set(sentence, claims);
      }

      return claimsMap;
    };

    const fetchPublicCheck = async () => {
      if (!checkId) return;

      try {
        const response = await client.api.agent.public[":checkId"].$get({
          param: { checkId: checkId as string },
        });

        if (!response.ok) {
          throw new Error("Failed to fetch check");
        }

        const data = await response.json();

        setClaims(extractClaims(data));
        setCheckData({
          title: data.metadata.title,
          textContent: data.metadata.text,
          completedAt: data.metadata.createdAt || new Date().toISOString(),
          user: data.metadata.user,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);
      }
    };

    fetchPublicCheck();
  }, [checkId]);

  if (loading) {
    return (
      <div className="container mx-auto max-w-6xl p-8">
        <div className="flex items-center justify-center py-16">
          <div className="animate-spin h-8 w-8 border-2 border-neutral-300 border-t-neutral-600 rounded-full" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto max-w-6xl p-8">
        <div className="text-center py-16">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
          <p className="text-neutral-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!checkData) {
    return (
      <div className="container mx-auto max-w-6xl p-8">
        <div className="text-center py-16">
          <h1 className="text-2xl font-bold text-neutral-800 mb-4">
            Not Found
          </h1>
          <p className="text-neutral-600">
            The requested check could not be found.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-6xl">
      <PageHeader />
      <AestheticBackground className="h-[80%] from-transparent! via-transparent! to-white!" />
      <div className="flex flex-col border border-b-0 rounded-t-2xl min-h-[94vh] bg-white">
        <div>
          <div className="space-y-4 p-4 pb-0">
            <h1 className="text-2xl font-bold text-neutral-900">
              {checkData?.title || "Fact Check Report"}
            </h1>
            <p className="text-neutral-600 text-sm">
              {checkData?.textContent || "No content available"}
            </p>
            <time className="text-neutral-700 text-sm">
              Verified {getRelativeTime(checkData?.completedAt || "")}
            </time>
          </div>

          <div className="p-4">
            <FactChecker
              claims={claims}
              isLoading={false}
              hasNoClaims={claims.size === 0}
            />
          </div>
        </div>

        <PageFooter />
      </div>
    </div>
  );
};

export default PublicCheckPage;
