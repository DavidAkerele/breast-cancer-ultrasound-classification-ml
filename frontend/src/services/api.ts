import type { DiagnosticResult, BatchResultItem, DatasetFileItem } from "../types";

const API_BASE_URL = "http://localhost:8000";

export async function checkHealth(): Promise<{ status: string; device: string; classes: string[] }> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) {
      // fallback to root if /health isn't mapped directly
      const rootRes = await fetch(`${API_BASE_URL}/`);
      if (rootRes.ok) return await rootRes.json();
      throw new Error(`Health check failed: ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.error("API Health error:", err);
    throw err;
  }
}

export async function predictScan(file: File, useClahe: boolean = true, model: string = "resnet50"): Promise<DiagnosticResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_clahe", useClahe ? "true" : "false");
  formData.append("model", model);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Diagnostic diagnosis failed on backend.");
  }

  return await res.json();
}

export async function predictBatchScans(files: File[], useClahe: boolean = true): Promise<{ results: BatchResultItem[] }> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });
  formData.append("use_clahe", useClahe ? "true" : "false");

  const res = await fetch(`${API_BASE_URL}/predict/batch`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Batch evaluation failed on backend.");
  }

  return await res.json();
}

export async function fetchDatasetScans(dataset: string = "busi", split: string = "val"): Promise<DatasetFileItem[]> {
  const res = await fetch(`${API_BASE_URL}/api/dataset/files?dataset=${dataset}&split=${split}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to load benchmark dataset files.");
  }
  return await res.json();
}

export async function fetchScanFileFromUrl(urlPath: string, filename: string): Promise<File> {
  const res = await fetch(`${API_BASE_URL}${urlPath}`);
  if (!res.ok) {
    throw new Error("Could not download dataset scan blob.");
  }
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || "image/png" });
}
