"use client";

import { SignOutButton, useUser } from "@clerk/nextjs";
import type { UserResource } from "@clerk/types";
import { Loader, LogOut, Plus, Settings, User2 } from "lucide-react";
import Link from "next/link";
import { Badge } from "../ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import {
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "../ui/sidebar";

const SidebarWrapper = ({ children }: React.PropsWithChildren) => (
  <SidebarFooter>
    <SidebarMenu>
      <SidebarMenuItem>{children}</SidebarMenuItem>
    </SidebarMenu>
  </SidebarFooter>
);

const MenuButton = (props: React.ComponentProps<typeof SidebarMenuButton>) => (
  <SidebarMenuButton
    className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
    size="lg"
    {...props}
  />
);

const UserAvatar = ({ user }: { user: any }) => {
  if (user?.imageUrl) {
    return (
      <img
        alt={user.fullName || "User"}
        className="h-8 w-8 rounded-full"
        src={user.imageUrl}
      />
    );
  }
  return <User2 />;
};

const LoadingState = () => (
  <SidebarWrapper>
    <MenuButton>
      <div className="flex size-8 items-center justify-center rounded-full bg-neutral-900">
        <Loader className="size-4 animate-spin text-white" />
      </div>
      <div className="grid flex-1 text-left text-sm leading-tight">
        <span className="truncate font-semibold">Loading...</span>
        <span className="truncate text-muted-foreground text-xs">
          Please wait
        </span>
      </div>
    </MenuButton>
  </SidebarWrapper>
);

const NotSignedInState = () => (
  <SidebarWrapper>
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <MenuButton>
          <User2 />
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="truncate font-semibold">Get Started</span>
            <span className="truncate text-xs">Auth & Settings</span>
          </div>
        </MenuButton>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
        side="bottom"
        sideOffset={4}
      >
        <DropdownMenuItem asChild>
          <Link href="/auth/sign-in">
            <User2 className="mr-2 size-4" />
            Sign In
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/auth/sign-up">
            <Plus className="mr-2 size-4" />
            Sign Up
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  </SidebarWrapper>
);

interface SignedInStateProps {
  user: UserResource;
}

const SignedInState = ({ user }: SignedInStateProps) => (
  <SidebarWrapper>
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <MenuButton>
          <UserAvatar user={user} />
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="truncate font-semibold">
              {user?.primaryEmailAddress?.emailAddress || "No email"}
            </span>
            <span className="truncate text-muted-foreground text-xs">
              Basic User
            </span>
          </div>
        </MenuButton>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
        side="bottom"
        sideOffset={4}
      >
        <DropdownMenuItem>
          <Settings className="mr-2 size-4" />
          Settings
          <Badge className="ml-auto text-xs" variant="secondary">
            Soon
          </Badge>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <SignOutButton redirectUrl="/auth/sign-in">
          <DropdownMenuItem>
            <LogOut className="mr-2 size-4" />
            Sign Out
          </DropdownMenuItem>
        </SignOutButton>
      </DropdownMenuContent>
    </DropdownMenu>
  </SidebarWrapper>
);

export const AppSidebarFooter = () => {
  const { user, isSignedIn, isLoaded } = useUser();

  if (!isLoaded) return <LoadingState />;
  if (!isSignedIn) return <NotSignedInState />;
  return <SignedInState user={user} />;
};
