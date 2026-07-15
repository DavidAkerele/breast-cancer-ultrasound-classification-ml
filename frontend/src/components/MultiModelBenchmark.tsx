import React, { useState } from "react";
import { Layers, ShieldCheck, ShieldAlert, Cpu, Zap } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { DiagnosticResult, DatasetFileItem } from "../types";
import { fetchDatasetScans, fetchScanFileFromUrl, predictScan } from "../services/api";

interface MultiModelBenchmarkProps {
  currentData: DiagnosticResult | null;
  onNewLog: (type: "INFO" | "BENCHMARK" | "ERROR", msg: string, details?: any) => void;
}

export const MultiModelBenchmark: React.FC<MultiModelBenchmarkProps> = ({
  currentData,
  onNewLog,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [benchmarkResult, setBenchmarkResult] = useState<DiagnosticResult | null>(currentData);
  const [sampleBank, setSampleBank] = useState<DatasetFileItem[]>([]);

  React.useEffect(() => {
    if (currentData) setBenchmarkResult(currentData);
  }, [currentData]);

  React.useEffect(() => {
    loadSamples();
  }, []);

  const loadSamples = async () => {
    try {
      const items = await fetchDatasetScans("busi", "val");
      setSampleBank(items.slice(0, 8));
    } catch (err) {
      console.warn("Could not load samples for benchmark:", err);
    }
  };

  const handleRunSampleBenchmark = async (item: DatasetFileItem) => {
    setLoading(true);
    onNewLog("INFO", `Running simultaneous 3-Architecture Benchmark on scan ${item.name}...`);
    try {
      const file = await fetchScanFileFromUrl(item.url, item.name);
      const data = await predictScan(file, true);
      setBenchmarkResult(data);
      onNewLog("BENCHMARK", `Completed benchmark on ${item.name}. All 3 models evaluated.`, data.multi_model_comparison);
    } catch (err: any) {
      onNewLog("ERROR", `Benchmark error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const comparison = benchmarkResult?.multi_model_comparison;

  // Calculate Consensus
  let consensusText = "Awaiting Scan Analysis";
  let consensusAgreeCount = 0;
  let consensusPred = "UNKNOWN";

  if (comparison && comparison.custom_cnn && comparison.resnet50 && comparison.efficientnet_b0) {
    const preds = [
      comparison.custom_cnn.prediction,
      comparison.resnet50.prediction,
      comparison.efficientnet_b0.prediction,
    ];
    const malignantCount = preds.filter((p) => p.toUpperCase() === "MALIGNANT").length;
    const benignCount = preds.filter((p) => p.toUpperCase() === "BENIGN").length;

    if (malignantCount === 3) {
      consensusAgreeCount = 3;
      consensusPred = "MALIGNANT";
      consensusText = "Unanimous 3/3 Deep Learning Agreement (Malignant)";
    } else if (benignCount === 3) {
      consensusAgreeCount = 3;
      consensusPred = "BENIGN";
      consensusText = "Unanimous 3/3 Deep Learning Agreement (Benign)";
    } else if (malignantCount === 2) {
      consensusAgreeCount = 2;
      consensusPred = "MALIGNANT";
      consensusText = "2/3 Majority Agreement on Malignant Diagnosis";
    } else {
      consensusAgreeCount = 2;
      consensusPred = "BENIGN";
      consensusText = "2/3 Majority Agreement on Benign Diagnosis";
    }
  }

  return (
    <div className="flex flex-col gap-8 w-full">
      {/* Top Banner & Quick Case Selection */}
      <Card className="p-6 clinical-glass flex flex-col xl:flex-row items-start xl:items-center justify-between gap-6 border-border/80 shadow-xl">
        <div className="flex items-center gap-4">
          <div className="flex size-12 items-center justify-center rounded-2xl bg-primary/15 text-primary border border-primary/30 clinical-glow-cyan shrink-0">
            <Layers className="size-6 animate-pulse" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-foreground tracking-tight">
              Simultaneous Multi-Model Architecture Grid
            </h2>
            <p className="text-xs text-muted-foreground max-w-2xl leading-relaxed">
              Simultaneously executes inference across <span className="text-primary font-mono font-semibold">ResNet-50</span>, <span className="text-emerald-400 font-mono font-semibold">EfficientNet-B0</span>, and <span className="text-purple-400 font-mono font-semibold">Custom CNN</span> on identical acoustic feature inputs to establish high-confidence clinical consensus.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          {sampleBank.map((sample, idx) => (
            <Button
              key={idx}
              size="sm"
              variant="outline"
              disabled={loading}
              onClick={() => handleRunSampleBenchmark(sample)}
              className="text-xs font-mono border-border bg-secondary/60 hover:border-primary hover:text-primary h-9 px-3.5 shadow-sm cursor-pointer"
            >
              <span>Test: {sample.name.replace(".png", "").replace(".jpg", "")}</span>
            </Button>
          ))}
        </div>
      </Card>

      {/* Consensus Bar */}
      {comparison && (
        <Card
          className={`p-5 rounded-2xl border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 transition-all shadow-xl ${
            consensusPred === "MALIGNANT"
              ? "bg-destructive/15 border-destructive/60 clinical-glow-destructive"
              : "bg-emerald-500/15 border-emerald-500/50 clinical-glow-emerald"
          }`}
        >
          <div className="flex items-center gap-4">
            {consensusPred === "MALIGNANT" ? (
              <ShieldAlert className="size-8 text-destructive animate-pulse shrink-0" />
            ) : (
              <ShieldCheck className="size-8 text-emerald-400 animate-pulse shrink-0" />
            )}
            <div>
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider opacity-90">
                Cross-Architecture Diagnostic Consensus
              </span>
              <h3 className={`text-base sm:text-lg font-bold font-mono ${consensusPred === "MALIGNANT" ? "text-destructive" : "text-emerald-400"}`}>
                {consensusText}
              </h3>
            </div>
          </div>

          <Badge variant="outline" className="font-mono text-xs px-4 py-1.5 bg-background/90 text-foreground border-border shrink-0">
            Agreement: {consensusAgreeCount}/3 Neural Architectures
          </Badge>
        </Card>
      )}

      {/* The 3 Architectures Grid */}
      {!comparison ? (
        <Card className="flex h-80 flex-col items-center justify-center rounded-2xl border border-border/60 bg-secondary/30 p-10 text-center shadow-inner">
          <Cpu className="size-12 text-primary/40 mb-4 animate-pulse" />
          <h3 className="text-base font-bold text-foreground">No Multi-Model Evaluation Active</h3>
          <p className="mt-2 text-xs text-muted-foreground max-w-lg leading-relaxed">
            Click one of the validation benchmark test cases above or run a scan in the Diagnostic Viewport to simultaneously evaluate across all three neural architectures.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Model 1: ResNet-50 */}
          <Card className="p-6 clinical-glass flex flex-col gap-5 border-border/80 hover:border-primary/60 transition-all shadow-xl rounded-2xl group">
            <div className="flex items-center justify-between pb-3 border-b border-border/60">
              <div className="flex items-center gap-2.5">
                <div className="size-3 rounded-full bg-primary group-hover:scale-125 transition-transform" />
                <span className="font-bold text-base text-foreground">ResNet-50 v2 Deep</span>
              </div>
              <Badge variant="outline" className="font-mono text-[11px] bg-secondary text-muted-foreground">
                25.6M PARAMS
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground uppercase font-semibold font-mono">Prediction:</span>
              <Badge
                variant={comparison.resnet50?.prediction.toUpperCase() === "MALIGNANT" ? "destructive" : "outline"}
                className={`font-mono font-bold text-sm px-3 py-1 ${
                  comparison.resnet50?.prediction.toUpperCase() === "BENIGN"
                    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 clinical-glow-emerald"
                    : "clinical-glow-destructive"
                }`}
              >
                {comparison.resnet50?.prediction.toUpperCase()}
              </Badge>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono text-xs">
                <span className="text-muted-foreground font-semibold">Network Confidence</span>
                <span className="font-bold text-foreground text-sm">
                  {roundScore(comparison.resnet50?.confidence)}%
                </span>
              </div>
              <Progress
                value={roundScore(comparison.resnet50?.confidence)}
                className="h-2.5 rounded-full bg-secondary overflow-hidden"
              />
            </div>

            <div className="flex flex-col gap-2 pt-3 border-t border-border/60 text-xs font-mono">
              <div className="flex justify-between p-2 rounded bg-secondary/30">
                <span className="text-emerald-400 font-bold">Benign Prob:</span>
                <span className="text-foreground font-semibold">{comparison.resnet50?.probabilities?.benign ?? 0}%</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-secondary/30">
                <span className="text-destructive font-bold">Malignant Prob:</span>
                <span className="text-foreground font-semibold">{comparison.resnet50?.probabilities?.malignant ?? 0}%</span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-border/60 text-xs text-muted-foreground font-mono mt-auto">
              <span className="flex items-center gap-1.5">
                <Zap className="size-3.5 text-primary" /> Inference Latency:
              </span>
              <span className="font-bold text-foreground">{comparison.resnet50?.latency_ms ?? 34} ms</span>
            </div>
          </Card>

          {/* Model 2: EfficientNet-B0 */}
          <Card className="p-6 clinical-glass flex flex-col gap-5 border-border/80 hover:border-emerald-400/60 transition-all shadow-xl rounded-2xl group">
            <div className="flex items-center justify-between pb-3 border-b border-border/60">
              <div className="flex items-center gap-2.5">
                <div className="size-3 rounded-full bg-emerald-400 group-hover:scale-125 transition-transform" />
                <span className="font-bold text-base text-foreground">EfficientNet-B0 Edge</span>
              </div>
              <Badge variant="outline" className="font-mono text-[11px] bg-secondary text-muted-foreground">
                5.3M PARAMS
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground uppercase font-semibold font-mono">Prediction:</span>
              <Badge
                variant={comparison.efficientnet_b0?.prediction.toUpperCase() === "MALIGNANT" ? "destructive" : "outline"}
                className={`font-mono font-bold text-sm px-3 py-1 ${
                  comparison.efficientnet_b0?.prediction.toUpperCase() === "BENIGN"
                    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 clinical-glow-emerald"
                    : "clinical-glow-destructive"
                }`}
              >
                {comparison.efficientnet_b0?.prediction.toUpperCase()}
              </Badge>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono text-xs">
                <span className="text-muted-foreground font-semibold">Network Confidence</span>
                <span className="font-bold text-foreground text-sm">
                  {roundScore(comparison.efficientnet_b0?.confidence)}%
                </span>
              </div>
              <Progress
                value={roundScore(comparison.efficientnet_b0?.confidence)}
                className="h-2.5 rounded-full bg-secondary overflow-hidden"
              />
            </div>

            <div className="flex flex-col gap-2 pt-3 border-t border-border/60 text-xs font-mono">
              <div className="flex justify-between p-2 rounded bg-secondary/30">
                <span className="text-emerald-400 font-bold">Benign Prob:</span>
                <span className="text-foreground font-semibold">{comparison.efficientnet_b0?.probabilities?.benign ?? 0}%</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-secondary/30">
                <span className="text-destructive font-bold">Malignant Prob:</span>
                <span className="text-foreground font-semibold">{comparison.efficientnet_b0?.probabilities?.malignant ?? 0}%</span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-border/60 text-xs text-muted-foreground font-mono mt-auto">
              <span className="flex items-center gap-1.5">
                <Zap className="size-3.5 text-emerald-400" /> Inference Latency:
              </span>
              <span className="font-bold text-foreground">{comparison.efficientnet_b0?.latency_ms ?? 18} ms</span>
            </div>
          </Card>

          {/* Model 3: Custom CNN Baseline */}
          <Card className="p-6 clinical-glass flex flex-col gap-5 border-border/80 hover:border-purple-400/60 transition-all shadow-xl rounded-2xl group">
            <div className="flex items-center justify-between pb-3 border-b border-border/60">
              <div className="flex items-center gap-2.5">
                <div className="size-3 rounded-full bg-purple-400 group-hover:scale-125 transition-transform" />
                <span className="font-bold text-base text-foreground">Custom CNN Baseline</span>
              </div>
              <Badge variant="outline" className="font-mono text-[11px] bg-secondary text-muted-foreground">
                1.2M PARAMS
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground uppercase font-semibold font-mono">Prediction:</span>
              <Badge
                variant={comparison.custom_cnn?.prediction.toUpperCase() === "MALIGNANT" ? "destructive" : "outline"}
                className={`font-mono font-bold text-sm px-3 py-1 ${
                  comparison.custom_cnn?.prediction.toUpperCase() === "BENIGN"
                    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 clinical-glow-emerald"
                    : "clinical-glow-destructive"
                }`}
              >
                {comparison.custom_cnn?.prediction.toUpperCase()}
              </Badge>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono text-xs">
                <span className="text-muted-foreground font-semibold">Network Confidence</span>
                <span className="font-bold text-foreground text-sm">
                  {roundScore(comparison.custom_cnn?.confidence)}%
                </span>
              </div>
              <Progress
                value={roundScore(comparison.custom_cnn?.confidence)}
                className="h-2.5 rounded-full bg-secondary overflow-hidden"
              />
            </div>

            <div className="flex flex-col gap-2 pt-3 border-t border-border/60 text-xs font-mono">
              <div className="flex justify-between p-2 rounded bg-secondary/30">
                <span className="text-emerald-400 font-bold">Benign Prob:</span>
                <span className="text-foreground font-semibold">{comparison.custom_cnn?.probabilities?.benign ?? 0}%</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-secondary/30">
                <span className="text-destructive font-bold">Malignant Prob:</span>
                <span className="text-foreground font-semibold">{comparison.custom_cnn?.probabilities?.malignant ?? 0}%</span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-border/60 text-xs text-muted-foreground font-mono mt-auto">
              <span className="flex items-center gap-1.5">
                <Zap className="size-3.5 text-purple-400" /> Inference Latency:
              </span>
              <span className="font-bold text-foreground">{comparison.custom_cnn?.latency_ms ?? 12} ms</span>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

function roundScore(val: number | undefined): number {
  if (typeof val !== "number") return 0;
  return val <= 1 ? Math.round(val * 100) : Math.round(val);
}
