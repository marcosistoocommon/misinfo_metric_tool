import Image from "next/image";
import { cn, extractDomain } from "@/lib/utils";

type FaviconProps = {
  url: string;
  size?: number;
  className?: string;
};

export const Favicon = ({ url, size = 14, className = "" }: FaviconProps) => {
  const domain = extractDomain(url);

  const handleError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;

    if (img.src.includes("/favicon.ico")) {
      img.src = `https://${domain}/favicon.ico`;
      return;
    }

    const parent = img.parentNode as HTMLElement;
    parent.innerHTML = `
      <div class="w-full h-full flex items-center justify-center bg-neutral-200 text-[8px] font-medium text-neutral-600 rounded-sm">
        ${domain.charAt(0).toUpperCase()}
      </div>
    `;
  };

  return (
    <div
      className={cn(
        "relative size-3.5 flex-shrink-0 overflow-hidden rounded-[2px]",
        className
      )}
    >
      <Image
        alt={`${domain} favicon`}
        className="h-full w-full object-cover"
        height={size}
        onError={handleError}
        src={`https://www.google.com/s2/favicons?domain=${domain}&sz=${size}`}
        width={size}
      />
    </div>
  );
};
