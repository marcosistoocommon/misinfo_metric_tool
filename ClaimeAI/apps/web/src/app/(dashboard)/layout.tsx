import { AppSidebar } from "@/components/nav/app-sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

const DashboardLayout = ({ children }: Readonly<React.PropsWithChildren>) => (
  <SidebarProvider className="bg-neutral-100!">
    <AppSidebar />
    <SidebarInset className="border-[0.5px] bg-neutral-50">
      {children}
    </SidebarInset>
  </SidebarProvider>
);

export default DashboardLayout;
