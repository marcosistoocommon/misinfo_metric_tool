import { EmptyState } from "@/components/empty-state";
import { InputSection } from "@/components/input-section";
import { MethodologyIndicators } from "@/components/methodology-indicators";

const HomePage = () => (
  <main
    className="mx-auto flex min-w-4xl max-w-4xl flex-grow flex-col items-center justify-center gap-3 p-6"
    id="main-content"
  >
    <EmptyState />
    <InputSection />
    <MethodologyIndicators />
  </main>
);

export default HomePage;
