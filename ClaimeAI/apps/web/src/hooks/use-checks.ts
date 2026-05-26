"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { CheckListItem } from "@/lib/check-utils";
import { client } from "@/lib/rpc";
import { useFactCheckerStore } from "@/lib/store";

const QUERY_KEY = ["checks"] as const;
const STALE_TIME = 1000 * 60 * 5;

export const useChecks = () => {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: async (): Promise<CheckListItem[]> => {
      const response = await client.api.agent.checks.$get();

      if (!response.ok) {
        throw new Error("Failed to fetch checks");
      }

      const data = await response.json();

      return data.checks.map((check) => ({
        ...check,
        createdAt: new Date(check.createdAt),
        updatedAt: new Date(check.updatedAt),
        completedAt: check.completedAt ? new Date(check.completedAt) : null,
        textPreview: check.textPreview || null,
      }));
    },
    staleTime: STALE_TIME,
    refetchOnWindowFocus: false,
  });
};

const createOptimisticCheck = (
  checkId: string,
  content: string
): CheckListItem => ({
  id: checkId,
  slug: checkId,
  title: null,
  status: "pending",
  createdAt: new Date(),
  updatedAt: new Date(),
  completedAt: null,
  textPreview: content.substring(0, 50),
});

export const useOptimisticCheckCreation = () => {
  const queryClient = useQueryClient();

  const addOptimisticCheck = (checkId: string, content: string) => {
    const optimisticCheck = createOptimisticCheck(checkId, content);
    queryClient.setQueryData<CheckListItem[]>(
      QUERY_KEY,
      (previousChecks = []) => [optimisticCheck, ...previousChecks]
    );
  };

  const titleGenerationMutation = useMutation({
    mutationFn: async ({
      content,
      checkId,
    }: {
      content: string;
      checkId: string;
    }) => {
      const response = await client.api.agent["generate-title"].$post({
        json: { content, checkId },
      });

      if (!response.ok) {
        throw new Error("Failed to generate title");
      }

      return await response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEY });
    },
  });

  return {
    addOptimisticCheck,
    generateTitle: titleGenerationMutation.mutate,
    isGeneratingTitle: titleGenerationMutation.isPending,
  };
};
