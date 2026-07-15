export interface Probabilities {
  benign?: number;
  malignant?: number;
  normal?: number;
  [key: string]: number | undefined;
}

export interface ModelComparisonResult {
  prediction: string;
  confidence: number;
  probabilities: Probabilities;
  latency_ms: number;
}

export interface NoiseAnalysis {
  dominant_type: string;
  description?: string;
  metrics: {
    snr_db: number;
    speckle_level: number;
    gaussian_level?: number;
    impulse_level?: number;
    entropy: number;
    mean_intensity: number;
    std_intensity: number;
  };
}

export interface NoiseSimulationResult {
  clean_prediction: string;
  clean_confidence: number;
  clean_image: string;
  noisy_prediction: string;
  noisy_confidence: number;
  noisy_image: string;
  explanation: string;
}

export interface ClinicalReport {
  birads: string;
  tissue_density: string;
  acoustic_shadowing: string;
  rationale: string;
}

export interface DiagnosticResult {
  prediction: string;
  confidence: number;
  probabilities: Probabilities;
  original_image: string;
  processed_image: string;
  spotlight_zoom_base64?: string;
  used_clahe: boolean;
  noise_analysis: NoiseAnalysis;
  multi_model_comparison: {
    custom_cnn?: ModelComparisonResult;
    resnet50?: ModelComparisonResult;
    efficientnet_b0?: ModelComparisonResult;
  };
  clinical_report: ClinicalReport;
}

export interface BatchResultItem {
  filename: string;
  prediction: string;
  confidence: number;
  original_image: string;
  processed_image: string;
  noise_analysis: NoiseAnalysis;
}

export interface DatasetFileItem {
  name: string;
  class: string;
  url: string;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  type: "INFO" | "DIAGNOSIS" | "BENCHMARK" | "BATCH" | "ERROR";
  message: string;
  details?: any;
}
