import React from "react";
import { Activity, Cpu, Layers, Terminal, ServerOff, Radio } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface HeaderNavProps {
  activeTab: string;
  onSelectTab: (tab: string) => void;
  apiConnected: boolean;
  deviceInfo: string;
}

export const HeaderNav: React.FC<HeaderNavProps> = ({
  activeTab,
  onSelectTab,
  apiConnected,
  deviceInfo,
}) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/90 backdrop-blur-md transition-all">
      <div className="w-full flex items-center justify-between px-4 py-3 sm:px-8 xl:px-12">
        {/* Brand & Workstation Identifier */}
        <div className="flex items-center gap-3.5">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/15 text-primary border border-primary/30 clinical-glow-cyan">
            <Radio className="size-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold tracking-widest text-primary uppercase">
                ONCO-DIAGNOSTIC ARCHITECTURE
              </span>
              <Badge variant="outline" className="h-4.5 px-2 text-[10px] font-mono border-primary/40 text-primary bg-primary/5">
                v4.2-PRO MAX
              </Badge>
            </div>
            <h1 className="text-sm font-semibold tracking-tight text-foreground sm:text-base">
              Deep Clinical Ultrasound Workstation
            </h1>
          </div>
        </div>

        {/* Tab-Driven Command Deck Navigation Bar */}
        <nav className="hidden md:flex items-center gap-1 rounded-xl bg-secondary/80 p-1.5 border border-border">
          <button
            onClick={() => onSelectTab("diagnostic")}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all cursor-pointer ${
              activeTab === "diagnostic"
                ? "bg-background text-primary shadow-sm border border-border font-semibold clinical-glow-cyan"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Activity className="size-4" />
            <span>Diagnostic Viewport</span>
          </button>

          <button
            onClick={() => onSelectTab("benchmark")}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all cursor-pointer ${
              activeTab === "benchmark"
                ? "bg-background text-primary shadow-sm border border-border font-semibold clinical-glow-cyan"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Layers className="size-4" />
            <span>Multi-Model Benchmark Grid</span>
          </button>

          <button
            onClick={() => onSelectTab("batch")}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all cursor-pointer ${
              activeTab === "batch"
                ? "bg-background text-primary shadow-sm border border-border font-semibold clinical-glow-cyan"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Terminal className="size-4" />
            <span>Batch & Audit Logs</span>
          </button>
        </nav>

        {/* System Telemetry Indicator */}
        <div className="flex items-center gap-3">
          {deviceInfo && (
            <div className="hidden lg:flex items-center gap-2 rounded-lg bg-secondary px-3 py-1.5 border border-border/80">
              <Cpu className="size-3.5 text-primary" />
              <span className="font-mono text-[11px] text-muted-foreground uppercase truncate max-w-[130px]">
                {deviceInfo}
              </span>
            </div>
          )}

          {apiConnected ? (
            <Badge className="flex items-center gap-2 bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-3 py-1.5 text-xs font-mono font-bold clinical-glow-emerald">
              <span className="size-2 rounded-full bg-emerald-400 animate-ping" />
              <span className="size-2 rounded-full bg-emerald-400 -ml-4" />
              <span>FastAPI Linked</span>
            </Badge>
          ) : (
            <Badge variant="destructive" className="flex items-center gap-2 px-3 py-1.5 text-xs font-mono font-bold clinical-glow-destructive">
              <ServerOff className="size-3.5" />
              <span>API Offline</span>
            </Badge>
          )}
        </div>
      </div>

      {/* Mobile Tab Bar switcher */}
      <div className="flex md:hidden border-t border-border bg-secondary/50 px-3 py-2 justify-around">
        <button
          onClick={() => onSelectTab("diagnostic")}
          className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium ${
            activeTab === "diagnostic" ? "bg-background text-primary font-semibold shadow-xs border border-border" : "text-muted-foreground"
          }`}
        >
          <Activity className="size-3.5" />
          <span>Viewport</span>
        </button>
        <button
          onClick={() => onSelectTab("benchmark")}
          className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium ${
            activeTab === "benchmark" ? "bg-background text-primary font-semibold shadow-xs border border-border" : "text-muted-foreground"
          }`}
        >
          <Layers className="size-3.5" />
          <span>Benchmark</span>
        </button>
        <button
          onClick={() => onSelectTab("batch")}
          className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium ${
            activeTab === "batch" ? "bg-background text-primary font-semibold shadow-xs border border-border" : "text-muted-foreground"
          }`}
        >
          <Terminal className="size-3.5" />
          <span>Logs</span>
        </button>
      </div>
    </header>
  );
};
