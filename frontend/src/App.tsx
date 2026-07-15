import { useState, useEffect } from "react";
import { HeaderNav } from "./components/HeaderNav";
import { DiagnosticViewport } from "./components/DiagnosticViewport";
import { NoiseLaboratory } from "./components/NoiseLaboratory";
import { MultiModelBenchmark } from "./components/MultiModelBenchmark";
import { BatchAnalytics } from "./components/BatchAnalytics";
import { checkHealth } from "./services/api";
import type { DiagnosticResult, AuditLogEntry } from "./types";

export function App() {
  const [activeTab, setActiveTab] = useState<string>("diagnostic");
  const [apiConnected, setApiConnected] = useState<boolean>(false);
  const [deviceInfo, setDeviceInfo] = useState<string>("DETECTING...");
  const [lastDiagnosticData, setLastDiagnosticData] = useState<DiagnosticResult | null>(null);
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);

  const appendLog = (type: "INFO" | "DIAGNOSIS" | "BENCHMARK" | "BATCH" | "ERROR", message: string, details?: any) => {
    const newLog: AuditLogEntry = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      type,
      message,
      details,
    };
    setLogs((prev) => [newLog, ...prev.slice(0, 49)]);
  };

  useEffect(() => {
    document.documentElement.classList.add("dark");

    const testConnection = async () => {
      try {
        const data = await checkHealth();
        setApiConnected(true);
        setDeviceInfo(data.device ? `DEVICE: ${data.device.toUpperCase()}` : "DEVICE: CPU/GPU");
        appendLog("INFO", "Connected to FastAPI Deep Learning Backend [Port 8000]. Models active.");
      } catch (err) {
        setApiConnected(false);
        setDeviceInfo("OFFLINE");
        appendLog("ERROR", "Could not establish heartbeat with FastAPI backend at http://localhost:8000.");
      }
    };

    testConnection();
    const interval = setInterval(testConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-screen max-h-screen overflow-hidden bg-background text-foreground flex flex-col font-sans selection:bg-primary/20 selection:text-primary">
      {/* Full-Width Command Deck Header Bar */}
      <HeaderNav
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        apiConnected={apiConnected}
        deviceInfo={deviceInfo}
      />

      {/* Full-Screen Spanning Workstation Body (Zero Page Scroll) */}
      <main className="flex-1 w-full px-4 sm:px-6 xl:px-8 py-3 flex flex-col min-h-0 overflow-hidden">
        {activeTab === "diagnostic" && (
          <DiagnosticViewport
            onNewLog={appendLog}
            onUpdateBenchmarkData={(data) => setLastDiagnosticData(data)}
          />
        )}

        {activeTab === "noise-lab" && (
          <NoiseLaboratory
            onNewLog={appendLog}
          />
        )}

        {activeTab === "benchmark" && (
          <MultiModelBenchmark
            currentData={lastDiagnosticData}
            onNewLog={appendLog}
          />
        )}

        {activeTab === "batch" && (
          <BatchAnalytics
            logs={logs}
            onNewLog={appendLog}
          />
        )}
      </main>

      {/* Compact Full-Width Footer System Status */}
      <footer className="shrink-0 border-t border-border/80 bg-background/90 py-2">
        <div className="w-full px-4 sm:px-6 xl:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted-foreground font-mono">
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-primary animate-pulse" />
            <span className="font-bold text-foreground">ANTIGRAVITY ONCO-DIAGNOSTIC WORKSTATION v4.2 PRO MAX</span>
          </div>
          <span className="opacity-80 truncate">
            Simultaneous Architectures: ResNet-50 v2 • EfficientNet-B0 • Custom CNN | Cohorts: BUSI / BrEaST / OASBUD
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
