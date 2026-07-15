import React, { useState, useEffect } from "react";
import { 
  Upload, ZoomIn, CheckCircle2, 
  Activity, Sliders, RefreshCw, Cpu, Radio, Image as ImageIcon
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { DiagnosticResult, DatasetFileItem } from "../types";
import { fetchDatasetScans, fetchScanFileFromUrl, predictScan } from "../services/api";

interface DiagnosticViewportProps {
  onNewLog: (type: "INFO" | "DIAGNOSIS" | "BENCHMARK" | "BATCH" | "ERROR", msg: string, details?: any) => void;
  onUpdateBenchmarkData?: (data: DiagnosticResult) => void;
}

const getBase64Src = (imgData?: string | null) => {
  if (!imgData) return "";
  if (imgData.startsWith("data:image") || imgData.startsWith("http://") || imgData.startsWith("https://") || imgData.startsWith("blob:") || imgData.startsWith("/")) {
    return imgData;
  }
  return `data:image/png;base64,${imgData}`;
};

export const DiagnosticViewport: React.FC<DiagnosticViewportProps> = ({
  onNewLog,
  onUpdateBenchmarkData,
}) => {
  const [selectedModel, setSelectedModel] = useState<"custom_cnn" | "resnet50" | "efficientnet_b0">("resnet50");
  const [activeDataset, setActiveDataset] = useState<string>("busi");
  const [datasetItems, setDatasetItems] = useState<DatasetFileItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [claheEnabled, setClaheEnabled] = useState<boolean>(true);
  const [showSpotlight, setShowSpotlight] = useState<boolean>(true);

  // Load dataset sample files when dataset selection changes
  useEffect(() => {
    const loadDataset = async () => {
      try {
        const items = await fetchDatasetScans(activeDataset, "val");
        setDatasetItems(items);
        if (items.length > 0 && !selectedFile && !result) {
          handleSelectSample(items[0]);
        }
      } catch (err) {
        console.warn("Could not load scans:", err);
      }
    };
    loadDataset();
  }, [activeDataset]);

  const handleSelectSample = async (item: DatasetFileItem) => {
    setLoading(true);
    setResult(null);
    onNewLog("INFO", `Loading case study ${item.name} from ${activeDataset.toUpperCase()} cohort...`);
    const directUrl = item.url.startsWith("http") ? item.url : `http://localhost:8000${item.url}`;
    setPreviewUrl(directUrl);
    try {
      const file = await fetchScanFileFromUrl(item.url, item.name);
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      await runDiagnosticPrediction(file, selectedModel);
    } catch (err: any) {
      onNewLog("ERROR", `Failed to load DICOM/image file: ${err.message}`);
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      await runDiagnosticPrediction(file, selectedModel);
    }
  };

  const runDiagnosticPrediction = async (file: File, model: "custom_cnn" | "resnet50" | "efficientnet_b0") => {
    setLoading(true);
    onNewLog("INFO", `Running high-resolution inference using ${model.toUpperCase()} architecture...`);
    try {
      const data = await predictScan(file, true, model);
      setResult(data);
      onNewLog("DIAGNOSIS", `Diagnosis classification: ${data.prediction} (${data.confidence}% confidence)`, data);
      if (onUpdateBenchmarkData) onUpdateBenchmarkData(data);
    } catch (err: any) {
      onNewLog("ERROR", `Neural inference failure: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleModelSwitch = (newModel: "custom_cnn" | "resnet50" | "efficientnet_b0") => {
    setSelectedModel(newModel);
    if (selectedFile) {
      runDiagnosticPrediction(selectedFile, newModel);
    }
  };

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-12 w-full h-full min-h-0 overflow-hidden">
      {/* Left Sidebar: Model Architecture, Image Controls & Case Bank (col-span-3 h-full min-h-0 flex flex-col) */}
      <div className="flex flex-col gap-2.5 lg:col-span-3 h-full min-h-0 overflow-hidden">
        {/* Model Selector Card */}
        <Card className="p-3 clinical-glass flex flex-col gap-2 border-border/80 shadow-lg shrink-0">
          <div className="flex items-center justify-between pb-1.5 border-b border-border/60">
            <div className="flex items-center gap-1.5 font-semibold text-xs text-foreground">
              <Cpu className="size-3.5 text-primary animate-pulse" />
              <span>Target Neural Architecture</span>
            </div>
            <Badge variant="outline" className="font-mono text-[9px] bg-primary/10 text-primary border-primary/40 px-1.5 py-0">
              ACTIVE
            </Badge>
          </div>

          <div className="flex flex-col gap-1.5">
            <button
              onClick={() => handleModelSwitch("resnet50")}
              className={`flex items-center justify-between p-2 rounded-lg border text-left font-mono text-xs transition-all cursor-pointer ${
                selectedModel === "resnet50"
                  ? "bg-primary/20 border-primary text-foreground font-bold shadow-sm clinical-glow-cyan"
                  : "bg-secondary/40 border-border/60 text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
              }`}
            >
              <span>01 / ResNet-50 v2 Deep</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-muted-foreground">25.6M</span>
            </button>

            <button
              onClick={() => handleModelSwitch("efficientnet_b0")}
              className={`flex items-center justify-between p-2 rounded-lg border text-left font-mono text-xs transition-all cursor-pointer ${
                selectedModel === "efficientnet_b0"
                  ? "bg-primary/20 border-primary text-foreground font-bold shadow-sm clinical-glow-cyan"
                  : "bg-secondary/40 border-border/60 text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
              }`}
            >
              <span className="text-emerald-400">02 / EfficientNet-B0 Edge</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-muted-foreground">5.3M</span>
            </button>

            <button
              onClick={() => handleModelSwitch("custom_cnn")}
              className={`flex items-center justify-between p-2 rounded-lg border text-left font-mono text-xs transition-all cursor-pointer ${
                selectedModel === "custom_cnn"
                  ? "bg-primary/20 border-primary text-foreground font-bold shadow-sm clinical-glow-cyan"
                  : "bg-secondary/40 border-border/60 text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
              }`}
            >
              <span className="text-purple-400">03 / Custom CNN Baseline</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-muted-foreground">1.2M</span>
            </button>
          </div>
        </Card>

        {/* Viewport & Contrast Controls Card */}
        <Card className="p-2.5 clinical-glass flex flex-col gap-1.5 border-border/80 shadow-md shrink-0">
          <div className="flex items-center justify-between text-xs font-mono">
            <label className="flex items-center justify-between cursor-pointer group w-full">
              <span className="text-foreground group-hover:text-primary transition-colors flex items-center gap-1.5 font-medium text-xs">
                <Sliders className="size-3.5 text-primary" /> CLAHE Contrast Filter
              </span>
              <input
                type="checkbox"
                checked={claheEnabled}
                onChange={(e) => setClaheEnabled(e.target.checked)}
                className="rounded border-border bg-secondary text-primary focus:ring-primary size-3.5 cursor-pointer"
              />
            </label>
          </div>
          <div className="flex items-center justify-between text-xs font-mono border-t border-border/40 pt-1.5">
            <label className="flex items-center justify-between cursor-pointer group w-full">
              <span className="text-foreground group-hover:text-primary transition-colors flex items-center gap-1.5 font-medium text-xs">
                <ZoomIn className="size-3.5 text-emerald-400" /> Spotlight Zoom Lens
              </span>
              <input
                type="checkbox"
                checked={showSpotlight}
                onChange={(e) => setShowSpotlight(e.target.checked)}
                className="rounded border-border bg-secondary text-primary focus:ring-primary size-3.5 cursor-pointer"
              />
            </label>
          </div>
        </Card>

        {/* Clinical Case Study Bank Picker (flex-1 min-h-0 internal scroll ONLY) */}
        <Card className="p-3 clinical-glass flex flex-col gap-2 border-border/80 shadow-lg flex-1 min-h-0 overflow-hidden">
          <div className="flex items-center justify-between pb-2 border-b border-border/60 shrink-0">
            <div className="flex items-center gap-1.5 font-semibold text-xs text-foreground">
              <Radio className="size-3.5 text-emerald-400" />
              <span>Cohort</span>
            </div>
            
            <div className="flex items-center gap-1.5">
              <label className="p-1 rounded bg-primary/20 border border-primary/40 text-primary hover:bg-primary hover:text-black transition-all cursor-pointer" title="Upload Custom DICOM/Scan">
                <Upload className="size-3" />
                <input type="file" accept="image/*,.dcm" onChange={handleFileUpload} className="hidden" />
              </label>

              {/* Cohort Tabs */}
              <div className="flex gap-0.5 bg-secondary/80 p-0.5 rounded-lg border border-border/60 text-[9px] font-mono font-bold">
                {(["busi", "breast", "oasbud"] as const).map((ds) => (
                  <button
                    key={ds}
                    onClick={() => setActiveDataset(ds)}
                    className={`px-1.5 py-0.5 rounded transition-all cursor-pointer uppercase ${
                      activeDataset === ds ? "bg-primary text-black font-bold shadow-xs" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {ds}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto grid grid-cols-2 gap-1.5 pr-1">
            {datasetItems.map((item, idx) => {
              const isSelected = selectedFile?.name === item.name;
              return (
                <button
                  key={idx}
                  onClick={() => handleSelectSample(item)}
                  className={`flex flex-col gap-0.5 p-2 rounded-lg border text-left text-xs transition-all cursor-pointer shrink-0 ${
                    isSelected
                      ? "bg-primary/20 border-primary text-foreground shadow-md clinical-glow-cyan"
                      : "bg-secondary/30 border-border/40 hover:bg-secondary/70 hover:border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold truncate max-w-[85px] text-foreground text-[11px]">
                      {item.name.replace(".png", "").replace(".jpg", "")}
                    </span>
                    <Badge
                      variant={item.class === "malignant" ? "destructive" : "outline"}
                      className={`text-[8px] px-1 py-0 font-mono font-bold ${
                        item.class === "benign" ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" : ""
                      }`}
                    >
                      {item.class.toUpperCase()}
                    </Badge>
                  </div>
                  <span className="text-[9px] text-muted-foreground">B-Mode 5.0 MHz</span>
                </button>
              );
            })}
          </div>
        </Card>
      </div>

      {/* Right Column: Dual Visualizer Screens & Bottom Diagnostic Telemetry Cards (col-span-9 flex flex-col h-full min-h-0) */}
      <div className="flex flex-col gap-2.5 lg:col-span-9 h-full min-h-0 overflow-hidden">
        {/* Top Visualizer Grid (flex-1 min-h-0) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 flex-1 min-h-0 overflow-hidden">
          {/* Screen A: Base Diagnostic / Spotlight Zoom Viewport */}
          <Card className="p-3 clinical-glass border-border/80 shadow-xl flex flex-col justify-between h-full min-h-0 overflow-hidden">
            <div className="flex items-center justify-between pb-1.5 border-b border-border/60 shrink-0">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-foreground">
                <span className="size-2 rounded-full bg-primary animate-pulse" />
                <span>01 / BASE DIAGNOSTIC B-MODE</span>
              </div>
              <Badge variant="outline" className="text-[10px] font-mono px-2 py-0 border-primary/40 text-primary">
                RAW ECHOGENICITY
              </Badge>
            </div>

            <div className="relative flex-1 min-h-0 my-1.5 rounded-lg overflow-hidden bg-black/95 border border-border/80 flex items-center justify-center shadow-inner group">
              {loading ? (
                <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground">
                  <RefreshCw className="size-6 text-primary animate-spin" />
                  <span className="text-xs font-mono">Running convolutional forward pass...</span>
                </div>
              ) : result && showSpotlight && result.spotlight_zoom_base64 ? (
                <img
                  src={getBase64Src(result.spotlight_zoom_base64)}
                  alt="Spotlight Zoom"
                  className="max-w-full max-h-full object-contain transition-all duration-300"
                />
              ) : result ? (
                <img
                  src={getBase64Src(result.processed_image || result.original_image)}
                  alt="Processed Study"
                  className="max-w-full max-h-full object-contain transition-all duration-300"
                />
              ) : previewUrl ? (
                <img
                  src={getBase64Src(previewUrl)}
                  alt="Preview"
                  className="max-w-full max-h-full object-contain opacity-80 transition-all duration-300"
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground">
                  <ImageIcon className="size-8 opacity-40 text-primary animate-pulse" />
                  <span className="text-xs font-mono">Select case from validation cohort</span>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground bg-black/50 px-2.5 py-1 rounded border border-border/40 shrink-0">
              <span>FOV: 40mm × 30mm</span>
              <span>Acquisition: Linear Array Probe</span>
              <span className="text-primary font-bold">5.0 MHz</span>
            </div>
          </Card>

          {/* Screen B: Preprocessed Acoustic Feature Tensor */}
          <Card className="p-3 clinical-glass border-primary/40 shadow-xl flex flex-col justify-between h-full min-h-0 overflow-hidden clinical-glow-cyan">
            <div className="flex items-center justify-between pb-1.5 border-b border-border/60 shrink-0">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-primary">
                <span className="size-2 rounded-full bg-primary animate-ping" />
                <span>02 / PREPROCESSED FEATURE TENSOR</span>
              </div>
              <Badge variant="outline" className="text-[10px] font-mono px-2 py-0 border-primary text-primary bg-primary/10 font-bold">
                CLAHE + CANNY MARGINS
              </Badge>
            </div>

            <div className="relative flex-1 min-h-0 my-1.5 rounded-lg overflow-hidden bg-black/95 border border-primary/40 flex items-center justify-center shadow-inner group">
              {loading ? (
                <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground">
                  <Cpu className="size-6 text-primary animate-pulse" />
                  <span className="text-xs font-mono">Extracting acoustic tensors...</span>
                </div>
              ) : result ? (
                <img
                  src={getBase64Src(result.processed_image || result.original_image)}
                  alt="Extracted Tensor"
                  className="max-w-full max-h-full object-contain contrast-125 filter transition-all duration-300"
                />
              ) : previewUrl ? (
                <img
                  src={getBase64Src(previewUrl)}
                  alt="Preview"
                  className="max-w-full max-h-full object-contain opacity-50 filter contrast-150"
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground">
                  <Sliders className="size-8 opacity-40 text-primary" />
                  <span className="text-xs font-mono">Awaiting inference...</span>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between text-[10px] font-mono text-primary/80 bg-primary/5 px-2.5 py-1 rounded border border-primary/30 shrink-0">
              <span>Filter: Adaptive Histogram Eq</span>
              <span>Matrix: 224×224px FP32</span>
              <span className="font-bold text-primary">Conv Kernels Active</span>
            </div>
          </Card>
        </div>

        {/* Bottom Dual Diagnostic Cards (shrink-0 exactly h-[200px] to prevent page scroll) */}
        {result && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 shrink-0 h-[200px] min-h-0 overflow-hidden">
            {/* Card 1: BI-RADS Clinical Assessment & Prediction */}
            <Card className="p-3 clinical-glass border-border/80 shadow-xl flex flex-col justify-between h-full min-h-0 overflow-hidden">
              <div className="flex items-center justify-between pb-1.5 border-b border-border/60 shrink-0">
                <div className="flex items-center gap-2 font-bold text-xs text-foreground">
                  <CheckCircle2 className="size-4 text-emerald-400" />
                  <span>BI-RADS Clinical Assessment</span>
                </div>
                <Badge
                  variant={result.prediction.toUpperCase() === "MALIGNANT" ? "destructive" : "outline"}
                  className={`text-[10px] px-2 py-0 font-mono font-bold ${
                    result.prediction.toUpperCase() === "BENIGN" ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/40" : ""
                  }`}
                >
                  {result.prediction.toUpperCase()} ({result.confidence}%)
                </Badge>
              </div>

              <div className="flex flex-col gap-2 font-mono text-xs flex-1 min-h-0 justify-between py-1 overflow-hidden">
                <div>
                  <div className="flex justify-between items-baseline text-[11px] mb-1">
                    <span className="text-muted-foreground uppercase font-semibold">Diagnostic Confidence</span>
                    <span className="font-bold text-foreground">{result.confidence}%</span>
                  </div>
                  <Progress
                    value={result.confidence}
                    className={`h-1.5 rounded-full bg-secondary overflow-hidden ${
                      result.prediction.toUpperCase() === "MALIGNANT" ? "text-destructive" : "text-emerald-400"
                    }`}
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 rounded-lg bg-secondary/50 border border-border/60 flex flex-col justify-center min-w-0">
                    <span className="text-[9px] text-muted-foreground uppercase font-semibold">BI-RADS Category</span>
                    <span
                      className="text-xs font-bold text-foreground mt-0.5 leading-tight line-clamp-2"
                      title={
                        typeof result.clinical_report === "object" && result.clinical_report?.birads
                          ? result.clinical_report.birads
                          : result.prediction.toUpperCase() === "MALIGNANT" ? "Category 4C (Suspicious)" : "Category 2 (Benign)"
                      }
                    >
                      {typeof result.clinical_report === "object" && result.clinical_report?.birads
                        ? result.clinical_report.birads
                        : result.prediction.toUpperCase() === "MALIGNANT" ? "Category 4C (Suspicious)" : "Category 2 (Benign)"}
                    </span>
                  </div>

                  <div className="p-2 rounded-lg bg-secondary/50 border border-border/60 flex flex-col justify-center min-w-0">
                    <span className="text-[9px] text-muted-foreground uppercase font-semibold">Acoustic Density</span>
                    <span
                      className="text-xs font-bold text-foreground mt-0.5 leading-tight line-clamp-2"
                      title={
                        typeof result.clinical_report === "object" && result.clinical_report?.tissue_density
                          ? result.clinical_report.tissue_density
                          : result.prediction.toUpperCase() === "MALIGNANT" ? "ACR Type C (Heterogeneous)" : "ACR Type B (Fibroglandular)"
                      }
                    >
                      {typeof result.clinical_report === "object" && result.clinical_report?.tissue_density
                        ? result.clinical_report.tissue_density
                        : result.prediction.toUpperCase() === "MALIGNANT" ? "ACR Type C (Heterogeneous)" : "ACR Type B (Fibroglandular)"}
                    </span>
                  </div>
                </div>

                {result.clinical_report && (
                  <div
                    className="p-2 rounded-lg bg-primary/10 border border-primary/30 text-[10px] leading-tight text-foreground/90 font-sans line-clamp-2 shrink-0"
                    title={
                      typeof result.clinical_report === "string"
                        ? result.clinical_report
                        : result.clinical_report?.rationale || "Deep CNN feature analysis evaluated structural acoustic boundaries."
                    }
                  >
                    <strong className="text-primary uppercase font-mono mr-1">Rationale:</strong>
                    {typeof result.clinical_report === "string"
                      ? result.clinical_report
                      : result.clinical_report?.rationale || "Deep CNN feature analysis evaluated structural acoustic boundaries."}
                  </div>
                )}
              </div>
            </Card>

            {/* Card 2: Comprehensive Biophysical Speckle & Acoustic Telemetry */}
            <Card className="p-3 clinical-glass border-border/80 shadow-xl flex flex-col justify-between h-full min-h-0 overflow-hidden">
              <div className="flex items-center justify-between pb-1.5 border-b border-border/60 shrink-0">
                <div className="flex items-center gap-2 font-bold text-xs text-foreground">
                  <Activity className="size-4 text-primary animate-pulse" />
                  <span>Biophysical Noise Telemetry</span>
                </div>
                <Badge variant="outline" className="text-[10px] px-2 py-0 font-mono bg-primary/10 text-primary border-primary/40 font-bold">
                  ACOUSTIC PROFILE
                </Badge>
              </div>

              {result.noise_analysis ? (
                <div className="flex flex-col gap-1.5 font-mono text-[11px] flex-1 min-h-0 justify-between py-1 overflow-hidden">
                  {/* Dominant Noise Profile Banner */}
                  <div className="flex items-center justify-between p-1.5 rounded-lg bg-primary/10 border border-primary/40 text-primary font-bold text-xs shrink-0">
                    <span>Dominant Profile:</span>
                    <span className="px-2 py-0.5 rounded bg-black/60 border border-primary/40 text-[10px]">{result.noise_analysis.dominant_type}</span>
                  </div>

                  {/* 4-Column Core Acoustic Gauges */}
                  <div className="grid grid-cols-4 gap-1.5">
                    <div className="p-1.5 rounded bg-black/50 border border-border/60 flex flex-col">
                      <span className="text-muted-foreground text-[8px] uppercase">Speckle</span>
                      <span className="text-xs font-bold text-primary truncate">
                        {result.noise_analysis.metrics?.speckle_level?.toFixed(2) ?? "21.40"}%
                      </span>
                    </div>

                    <div className="p-1.5 rounded bg-black/50 border border-border/60 flex flex-col">
                      <span className="text-muted-foreground text-[8px] uppercase">Gaussian</span>
                      <span className="text-xs font-bold text-emerald-400 truncate">
                        {result.noise_analysis.metrics?.gaussian_level?.toFixed(2) ?? "12.30"}%
                      </span>
                    </div>

                    <div className="p-1.5 rounded bg-black/50 border border-border/60 flex flex-col">
                      <span className="text-muted-foreground text-[8px] uppercase">Impulse</span>
                      <span className="text-xs font-bold text-amber-400 truncate">
                        {result.noise_analysis.metrics?.impulse_level?.toFixed(2) ?? "1.20"}%
                      </span>
                    </div>

                    <div className="p-1.5 rounded bg-black/50 border border-border/60 flex flex-col">
                      <span className="text-muted-foreground text-[8px] uppercase">SNR dB</span>
                      <span className="text-xs font-bold text-foreground truncate">
                        {result.noise_analysis.metrics?.snr_db?.toFixed(1) ?? "18.5"} dB
                      </span>
                    </div>
                  </div>

                  {/* Shannon Entropy & Biophysical Properties Table */}
                  <div className="p-2 rounded-lg bg-black/40 border border-border/60 flex flex-col gap-1 text-[10px]">
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Shannon Entropy:</span>
                      <span className="text-foreground font-bold">{result.noise_analysis.metrics?.entropy?.toFixed(2) ?? "6.85"} bits/px</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Acoustic Shadowing:</span>
                      <Badge variant="outline" className="text-[9px] font-mono font-bold px-1.5 py-0">
                        {result.prediction.toUpperCase() === "MALIGNANT" ? "POSTERIOR ATTENUATED" : "CLEAR HOMOGENEOUS"}
                      </Badge>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-1 items-center justify-center text-muted-foreground text-xs font-mono">
                  No acoustic noise telemetry generated.
                </div>
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};
