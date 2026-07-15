import React, { useState } from "react";
import { 
  Terminal, Upload, Play, RefreshCw, BarChart3, Database
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { BatchResultItem, AuditLogEntry } from "../types";
import { fetchDatasetScans, fetchScanFileFromUrl, predictBatchScans } from "../services/api";

interface BatchAnalyticsProps {
  logs: AuditLogEntry[];
  onNewLog: (type: "INFO" | "BATCH" | "ERROR", msg: string, details?: any) => void;
}

export const BatchAnalytics: React.FC<BatchAnalyticsProps> = ({
  logs,
  onNewLog,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [batchResults, setBatchResults] = useState<BatchResultItem[]>([]);
  const [filterMode, setFilterMode] = useState<"ALL" | "BENIGN" | "MALIGNANT">("ALL");

  const handleRunValidationSuite = async () => {
    setLoading(true);
    onNewLog("INFO", "Initiating high-throughput Batch Validation Suite on 8 clinical cases from BUSI study bank...");
    try {
      const items = await fetchDatasetScans("busi", "val");
      const testSubset = items.slice(0, 8);
      
      const files: File[] = [];
      for (const item of testSubset) {
        const file = await fetchScanFileFromUrl(item.url, item.name);
        files.push(file);
      }

      const response = await predictBatchScans(files, true);
      setBatchResults(response.results);
      onNewLog("BATCH", `Successfully analyzed ${response.results.length} patient scans in batch mode.`, response.results);
    } catch (err: any) {
      onNewLog("ERROR", `Batch execution failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const fileList = Array.from(e.target.files);
    setLoading(true);
    onNewLog("INFO", `Uploading ${fileList.length} scans for batch inference...`);
    try {
      const response = await predictBatchScans(fileList, true);
      setBatchResults(response.results);
      onNewLog("BATCH", `Processed ${response.results.length} uploaded files successfully.`, response.results);
    } catch (err: any) {
      onNewLog("ERROR", `Batch upload inference failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const totalAnalyzed = batchResults.length;
  const benignCount = batchResults.filter((r) => r.prediction === "BENIGN").length;
  const malignantCount = batchResults.filter((r) => r.prediction === "MALIGNANT").length;
  const avgConfidence = totalAnalyzed > 0 
    ? (batchResults.reduce((acc, curr) => acc + curr.confidence, 0) / totalAnalyzed).toFixed(1)
    : "0.0";

  const filteredResults = batchResults.filter((r) => {
    if (filterMode === "ALL") return true;
    return r.prediction === filterMode;
  });

  return (
    <div className="grid grid-cols-1 gap-8 lg:grid-cols-12 w-full">
      {/* Left Area: Batch Trigger & KPI Summary Cards (col-span-7) */}
      <div className="flex flex-col gap-6 lg:col-span-7">
        <Card className="p-6 clinical-glass flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 border-border/80 shadow-xl">
          <div className="flex items-center gap-4">
            <div className="flex size-12 items-center justify-center rounded-2xl bg-primary/15 text-primary border border-primary/30 clinical-glow-cyan shrink-0">
              <Database className="size-6" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-foreground tracking-tight">
                High-Throughput Batch Evaluation Engine
              </h2>
              <p className="text-xs text-muted-foreground mt-1">
                Run batch validation suites across BUSI / BrEaST cohorts or upload patient scan packages.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5 shrink-0">
            <Button
              size="sm"
              onClick={handleRunValidationSuite}
              disabled={loading}
              className="bg-primary hover:bg-primary/90 text-primary-foreground font-medium text-xs clinical-glow-cyan px-4 h-10 shadow-md cursor-pointer"
            >
              {loading ? (
                <>
                  <RefreshCw className="size-4 mr-2 animate-spin" />
                  <span>Processing Suite...</span>
                </>
              ) : (
                <>
                  <Play className="size-4 mr-2" />
                  <span>Run BUSI Suite (8 Scans)</span>
                </>
              )}
            </Button>

            <label className="inline-flex items-center justify-center rounded-lg text-xs font-medium border border-border bg-secondary/80 hover:bg-secondary text-foreground px-4 h-10 cursor-pointer transition-all shadow-sm">
              <Upload className="size-4 mr-2 text-primary" />
              <span>Upload Package</span>
              <input type="file" multiple accept="image/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>
        </Card>

        {/* KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card className="p-4 clinical-glass border-border/80 flex flex-col justify-between shadow-md">
            <span className="text-[11px] font-mono text-muted-foreground uppercase font-semibold">Total Processed</span>
            <span className="text-2xl font-bold font-mono text-foreground mt-2">{totalAnalyzed} Scans</span>
          </Card>

          <Card className="p-4 clinical-glass border-border/80 flex flex-col justify-between shadow-md">
            <span className="text-[11px] font-mono text-emerald-400 uppercase font-semibold">Benign Scans</span>
            <span className="text-2xl font-bold font-mono text-emerald-400 mt-2">{benignCount}</span>
          </Card>

          <Card className="p-4 clinical-glass border-border/80 flex flex-col justify-between shadow-md">
            <span className="text-[11px] font-mono text-destructive uppercase font-semibold">Malignant Scans</span>
            <span className="text-2xl font-bold font-mono text-destructive mt-2">{malignantCount}</span>
          </Card>

          <Card className="p-4 clinical-glass border-border/80 flex flex-col justify-between shadow-md">
            <span className="text-[11px] font-mono text-primary uppercase font-semibold">Avg Confidence</span>
            <span className="text-2xl font-bold font-mono text-primary mt-2">{avgConfidence}%</span>
          </Card>
        </div>

        {/* Batch Results Table */}
        <Card className="p-6 clinical-glass flex flex-col gap-5 border-border/80 shadow-xl flex-1">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-border/60">
            <div className="flex items-center gap-2.5 text-foreground font-bold text-base">
              <BarChart3 className="size-5 text-primary" />
              <span>Batch Diagnostic Ledger</span>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setFilterMode("ALL")}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                  filterMode === "ALL" ? "bg-primary text-primary-foreground shadow-sm" : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                }`}
              >
                ALL ({totalAnalyzed})
              </button>
              <button
                onClick={() => setFilterMode("BENIGN")}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                  filterMode === "BENIGN" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40" : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                }`}
              >
                BENIGN ({benignCount})
              </button>
              <button
                onClick={() => setFilterMode("MALIGNANT")}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                  filterMode === "MALIGNANT" ? "bg-destructive/20 text-destructive border border-destructive/40" : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                }`}
              >
                MALIGNANT ({malignantCount})
              </button>
            </div>
          </div>

          {filteredResults.length === 0 ? (
            <div className="flex h-64 flex-col items-center justify-center text-center text-muted-foreground font-mono">
              <span className="text-xs">No scan records available in this view.</span>
            </div>
          ) : (
            <div className="overflow-x-auto max-h-96 overflow-y-auto pr-1">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-border/60 text-[11px] font-mono text-muted-foreground uppercase">
                    <th className="py-3 px-4 font-semibold">Patient Study ID</th>
                    <th className="py-3 px-4 font-semibold">Diagnostic Assessment</th>
                    <th className="py-3 px-4 font-semibold">Confidence</th>
                    <th className="py-3 px-4 font-semibold">Noise Profile</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40 font-mono">
                  {filteredResults.map((row, idx) => (
                    <tr key={idx} className="hover:bg-secondary/40 transition-colors">
                      <td className="py-3 px-4 font-semibold text-foreground truncate max-w-[200px]">
                        {row.filename}
                      </td>
                      <td className="py-3 px-4">
                        <Badge
                          variant={row.prediction === "MALIGNANT" ? "destructive" : "outline"}
                          className={`text-xs px-2.5 py-0.5 font-bold ${
                            row.prediction === "BENIGN" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40" : ""
                          }`}
                        >
                          {row.prediction}
                        </Badge>
                      </td>
                      <td className="py-3 px-4 text-foreground font-bold text-sm">
                        {row.confidence}%
                      </td>
                      <td className="py-3 px-4 text-muted-foreground text-xs">
                        {row.noise_analysis?.dominant_type || "Speckle Profile"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* Right Column: Real-time Audit Log Terminal (`col-span-5`) */}
      <div className="flex flex-col gap-6 lg:col-span-5">
        <Card className="p-6 clinical-glass flex flex-col gap-4 border-border/80 h-full max-h-[740px] shadow-2xl">
          <div className="flex items-center justify-between pb-3 border-b border-border/60">
            <div className="flex items-center gap-2.5 text-foreground font-bold text-base">
              <Terminal className="size-5 text-primary animate-pulse" />
              <span>Live Clinical Audit & Inference Terminal</span>
            </div>
            <Badge variant="outline" className="font-mono text-[11px] bg-secondary text-muted-foreground">
              LOG LEVEL: DEBUG
            </Badge>
          </div>

          <div className="flex-1 overflow-y-auto rounded-xl bg-black/90 p-4 font-mono text-xs border border-border/60 flex flex-col gap-3 max-h-[610px] shadow-inner">
            {logs.length === 0 ? (
              <div className="text-muted-foreground text-center py-12">
                System telemetry standby. Execute diagnostic actions to monitor neural network output.
              </div>
            ) : (
              logs.map((log) => (
                <div key={log.id} className="flex flex-col gap-1.5 border-b border-border/30 pb-2.5 last:border-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-muted-foreground opacity-70 font-semibold">[{log.timestamp}]</span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                        log.type === "ERROR"
                          ? "bg-destructive/30 text-destructive"
                          : log.type === "DIAGNOSIS"
                          ? "bg-primary/20 text-primary border border-primary/40"
                          : log.type === "BENCHMARK"
                          ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                          : "bg-secondary text-muted-foreground"
                      }`}
                    >
                      {log.type}
                    </span>
                  </div>
                  <span className="text-foreground/95 pl-1 leading-snug">{log.message}</span>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};
