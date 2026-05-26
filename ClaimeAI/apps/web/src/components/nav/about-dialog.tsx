import { Info } from "lucide-react";
import { Badge } from "../ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import { SidebarMenuButton } from "../ui/sidebar";

const FEATURES = [
  "Automated claim extraction using the Claimify methodology",
  "Evidence-based verification through web search",
  "Detailed reporting with source citations",
  "Real-time processing and analysis",
];

const BADGES = ["AI-Powered", "Research-Based", "Open Source"];

export const AboutDialog = () => {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <SidebarMenuButton tooltip="Learn about our fact-checking system">
          <Info />
          <span>About ClaimeAI</span>
        </SidebarMenuButton>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[525px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            About ClaimeAI
          </DialogTitle>
          <DialogDescription className="space-y-4 text-left">
            <p>
              ClaimeAI is an advanced AI-powered fact-checking system that helps
              you verify the accuracy of textual claims using cutting-edge
              research methodologies.
            </p>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Key Features:</h4>
              <ul className="list-inside list-disc space-y-1 text-sm">
                {FEATURES.map((feature) => (
                  <li key={feature}>{feature}</li>
                ))}
              </ul>
            </div>
            <div className="flex gap-2 pt-2">
              {BADGES.map((badge) => (
                <Badge key={badge} variant="secondary">
                  {badge}
                </Badge>
              ))}
            </div>
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
};
