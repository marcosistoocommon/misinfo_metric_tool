"use client";

import { BookOpen, ExternalLink } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useRef } from "react";

import { Skeleton } from "@/components/ui/skeleton";
import { useChecks } from "@/hooks/use-checks";
import { getCheckGroupsForDisplay, groupChecksByTime } from "@/lib/check-utils";
import { cn } from "@/lib/utils";
import { HomeIcon, type HomeIconHandle } from "../ui/icons/home";
import { LoaderIcon } from "../ui/icons/loader";
import type { PlusIconHandle } from "../ui/icons/plus";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "../ui/sidebar";
import { AboutDialog } from "./about-dialog";
import { HowItWorksDialog } from "./how-it-works-dialog";
import { AppSidebarFooter } from "./sidebar-footer";
import Image from "next/image";

const NAVIGATION_ITEMS = [
  {
    title: "New Check",
    url: "/",
    icon: HomeIcon,
    description: "Main fact-checking interface",
  },
];

const SKELETON_GROUPS = [
  { itemCount: 3, labelWidth: "w-16", key: "recent" },
  { itemCount: 5, labelWidth: "w-20", key: "this-week" },
  { itemCount: 5, labelWidth: "w-20", key: "last-week" },
  { itemCount: 5, labelWidth: "w-20", key: "this-month" },
  { itemCount: 5, labelWidth: "w-20", key: "older" },
];

interface SkeletonGroupProps {
  itemCount: number;
  labelWidth: string;
  groupKey: string;
}

const SkeletonGroup = ({
  itemCount,
  labelWidth,
  groupKey,
}: SkeletonGroupProps) => (
  <SidebarGroup>
    <SidebarGroupLabel>
      <Skeleton className={cn("h-4 bg-neutral-200", labelWidth)} />
    </SidebarGroupLabel>
    <SidebarGroupContent>
      <SidebarMenu>
        {Array.from(
          { length: itemCount },
          (_, itemIndex) => `${groupKey}-${itemIndex}`
        ).map((uniqueKey) => (
          <SidebarMenuItem key={`skeleton-${uniqueKey}`}>
            <SidebarMenuButton disabled>
              <Skeleton className="h-8 w-full border bg-neutral-200" />
            </SidebarMenuButton>
          </SidebarMenuItem>
        ))}
      </SidebarMenu>
    </SidebarGroupContent>
  </SidebarGroup>
);

interface CheckGroupProps {
  checkGroup: ReturnType<typeof getCheckGroupsForDisplay>[0];
  currentPath: string;
}

const getCheckDisplayTitle = (
  check: CheckGroupProps["checkGroup"]["checks"][0]
) => {
  if (check.title) return check.title;
  if (check.textPreview) return `${check.textPreview}...`;
  return "Untitled Check";
};

const CheckGroup = ({ checkGroup, currentPath }: CheckGroupProps) => (
  <SidebarGroup>
    <SidebarGroupLabel>{checkGroup.label}</SidebarGroupLabel>
    <SidebarGroupContent>
      <SidebarMenu>
        {checkGroup.checks.map((check) => {
          const displayTitle = getCheckDisplayTitle(check);
          const isLoadingTitle = !check.title && check.status === "pending";
          const isActive = currentPath === `/checks/${check.slug}`;

          return (
            <SidebarMenuItem key={check.id}>
              <SidebarMenuButton
                asChild
                isActive={isActive}
                tooltip={displayTitle}
              >
                <Link href={`/checks/${check.slug}`}>
                  <span className="truncate">
                    {isLoadingTitle ? "Loading..." : displayTitle}
                  </span>
                  {isLoadingTitle && (
                    <LoaderIcon className="ml-auto shrink-0" />
                  )}
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          );
        })}
      </SidebarMenu>
    </SidebarGroupContent>
  </SidebarGroup>
);

interface NavigationItemsProps {
  currentPath: string;
  iconReferences: React.RefObject<HomeIconHandle | PlusIconHandle | null>[];
  onIconAnimationStart: (index: number) => void;
  onIconAnimationStop: (index: number) => void;
}

const NavigationItems = ({
  currentPath,
  iconReferences,
  onIconAnimationStart,
  onIconAnimationStop,
}: NavigationItemsProps) => {
  return (
    <SidebarGroup className="sticky top-0 z-10 bg-neutral-50 py-0">
      <SidebarGroupLabel>Fact Checking</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {NAVIGATION_ITEMS.map(({ icon: IconComponent, ...item }, index) => {
            const isActive = currentPath === item.url;

            return (
              <SidebarMenuItem
                key={item.title}
                onMouseEnter={() => onIconAnimationStart(index)}
                onMouseLeave={() => onIconAnimationStop(index)}
              >
                <SidebarMenuButton
                  asChild
                  isActive={isActive}
                  tooltip={item.description}
                >
                  <Link href={item.url}>
                    <IconComponent ref={iconReferences[index]} />
                    <span>{item.title}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            );
          })}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
};

const LearnMoreSection = () => (
  <SidebarGroup className="sticky bottom-0 z-10 mt-auto bg-neutral-50 py-0">
    <SidebarGroupLabel>Learn More</SidebarGroupLabel>
    <SidebarGroupContent>
      <SidebarMenu>
        <SidebarMenuItem>
          <AboutDialog />
        </SidebarMenuItem>

        <SidebarMenuItem>
          <HowItWorksDialog />
        </SidebarMenuItem>

        <SidebarMenuItem>
          <SidebarMenuButton
            asChild
            tooltip="View technical documentation on GitHub"
          >
            <Link
              className="flex items-center gap-2"
              href="https://github.com/BharathxD/claime-ai"
              target="_blank"
            >
              <BookOpen />
              <span>Documentation</span>
              <ExternalLink
                className="ml-auto size-2.5 opacity-70"
                strokeWidth={1.5}
              />
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroupContent>
  </SidebarGroup>
);

export const AppSidebar = () => {
  const currentPath = usePathname();
  const { data: checksData, isLoading: isLoadingChecks } = useChecks();

  const homeIconRef = useRef<HomeIconHandle>(null);
  const plusIconRef = useRef<PlusIconHandle>(null);
  const iconReferences = [homeIconRef, plusIconRef];

  const handleIconAnimationStart = (iconIndex: number) => {
    iconReferences[iconIndex]?.current?.startAnimation();
  };

  const handleIconAnimationStop = (iconIndex: number) => {
    iconReferences[iconIndex]?.current?.stopAnimation();
  };

  const checkGroups = useMemo(() => {
    if (!checksData || isLoadingChecks) return [];
    const timeGroupedChecks = groupChecksByTime(checksData);
    return getCheckGroupsForDisplay(timeGroupedChecks);
  }, [checksData, isLoadingChecks]);

  const hasChecks = checksData && checksData.length > 0;

  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild size="lg" className="gap-1 pl-0">
              <Link href="/">
                <Image
                  alt="ClaimeAI"
                  className="size-10"
                  height={40}
                  src="/logo.svg"
                  width={40}
                />
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">ClaimeAI</span>
                  <span className="truncate text-xs">AI Fact Checker</span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent className="no-scrollbar flex flex-col gap-0">
        <NavigationItems
          currentPath={currentPath}
          iconReferences={iconReferences}
          onIconAnimationStart={handleIconAnimationStart}
          onIconAnimationStop={handleIconAnimationStop}
        />

        {isLoadingChecks &&
          SKELETON_GROUPS.map((group) => (
            <SkeletonGroup
              groupKey={group.key}
              itemCount={group.itemCount}
              key={group.key}
              labelWidth={group.labelWidth}
            />
          ))}

        {!isLoadingChecks &&
          hasChecks &&
          checkGroups.map((group) => (
            <CheckGroup
              checkGroup={group}
              currentPath={currentPath}
              key={group.label}
            />
          ))}

        <LearnMoreSection />
      </SidebarContent>

      <AppSidebarFooter />
    </Sidebar>
  );
};
