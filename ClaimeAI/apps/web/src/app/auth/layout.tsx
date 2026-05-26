import { AestheticBackground } from "@/components/ui/aesthetic-background";

const AuthLayout = ({ children }: Readonly<React.PropsWithChildren>) => (
  <div className="flex h-screen flex-col items-center justify-center">
    {children}
    <AestheticBackground className="h-full from-white/0! via-white/0!" />
  </div>
);

export default AuthLayout;
