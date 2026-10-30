import { MotionConfig } from "motion/react";

import { About } from "./components/about/About";
import { DatasetSection } from "./components/dataset/DatasetSection";
import { LiveDemo } from "./components/demo/LiveDemo";
import { Hero } from "./components/hero/Hero";
import { HowItWorks } from "./components/how/HowItWorks";
import { DemoBanner } from "./components/layout/DemoBanner";
import { Footer } from "./components/layout/Footer";
import { ProjectLinks } from "./components/links/ProjectLinks";
import { Navbar } from "./components/layout/Navbar";
import { VideoProvider } from "./components/layout/VideoDialog";
import { ModelPerformance } from "./components/performance/ModelPerformance";
import { SiteDataProvider } from "./hooks/useSiteData";

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <SiteDataProvider>
        <VideoProvider>
          <a href="#main" className="skip-link">
            Skip to content
          </a>
          <DemoBanner />
          <Navbar />
          <main id="main">
            <Hero />
            <LiveDemo />
            <HowItWorks />
            <ModelPerformance />
            <DatasetSection />
            <ProjectLinks />
            <About />
          </main>
          <Footer />
        </VideoProvider>
      </SiteDataProvider>
    </MotionConfig>
  );
}
