/* OncoVision Research — dependency-light interface controller. */
document.addEventListener("DOMContentLoaded", () => {
    const byId = (id) => document.getElementById(id);
    const views = [...document.querySelectorAll(".view")];
    const navButtons = [...document.querySelectorAll("[data-view]")];
    const modelSelect = byId("modelSelect");
    const cropSelect = byId("cropSelect");
    const claheToggle = byId("claheToggle");
    const sidebar = byId("sidebar");
    const menuButton = byId("menuButton");

    const viewCopy = {
        workstation: ["Single-image inspection", "Trace one image through preprocessing, classification, and visual explanation."],
        noise: ["Noise experiment", "Compare a seeded synthetic perturbation across documented preprocessing paths."],
        batch: ["Batch evaluation", "Review multiple model outputs without assigning clinical priority."],
        evidence: ["Evidence record", "Read generated metrics together with the subject-level split audit."],
        dataset: ["Dataset explorer", "Inspect local image folders and their current labels."],
        documentation: ["Documentation", "Read the submission-facing project record directly from source."],
    };

    const state = {
        currentFile: null,
        currentGroundTruth: null,
        inference: null,
        imageObjectUrl: null,
        noiseFile: null,
        batchResults: [],
        batchLabels: new Map(),
        evidenceLoaded: false,
    };

    let toastTimer;
    function toast(message) {
        const element = byId("toast");
        element.textContent = message;
        element.classList.add("is-visible");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => element.classList.remove("is-visible"), 3200);
    }

    function showError(message) {
        byId("errorDialogMessage").textContent = message;
        const dialog = byId("errorDialog");
        if (typeof dialog.showModal === "function") dialog.showModal();
        else window.alert(message);
    }

    function setSystemStatus(message, status = "loading") {
        byId("systemStatus").textContent = message;
        byId("systemStatusDot").className = `status-dot ${status}`;
    }

    async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let body;
        try { body = await response.json(); }
        catch { body = { detail: `Request failed with HTTP ${response.status}.` }; }
        if (!response.ok) throw new Error(body.detail || `Request failed with HTTP ${response.status}.`);
        return body;
    }

    function switchView(name, updateHash = true) {
        const next = viewCopy[name] ? name : "workstation";
        views.forEach((view) => {
            const active = view.id === `view-${next}`;
            view.hidden = !active;
            view.classList.toggle("is-active", active);
        });
        navButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.view === next));
        byId("viewTitle").textContent = viewCopy[next][0];
        byId("viewDescription").textContent = viewCopy[next][1];
        sidebar.classList.remove("is-open");
        menuButton.setAttribute("aria-expanded", "false");
        if (updateHash) history.replaceState(null, "", `#${next}`);
        if (next === "evidence") loadEvidence();
        if (next === "dataset") loadDataset();
        if (next === "documentation" && !byId("documentViewer").dataset.loaded) loadDocument("DISSERTATION.md");
        byId("mainContent").focus({ preventScroll: true });
    }

    navButtons.forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
    document.querySelectorAll("[data-open-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.openView)));
    menuButton.addEventListener("click", () => {
        const open = sidebar.classList.toggle("is-open");
        menuButton.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && sidebar.classList.contains("is-open")) {
            sidebar.classList.remove("is-open");
            menuButton.setAttribute("aria-expanded", "false");
            menuButton.focus();
        }
    });

    async function checkHealth() {
        setSystemStatus("Checking model…", "loading");
        try {
            const health = await fetchJson("/health");
            (health.models || []).forEach(({ name, ready }) => {
                const option = modelSelect.querySelector(`option[value="${name}"]`);
                if (option) option.disabled = !ready;
            });
            if (health.model_ready) {
                modelSelect.value = health.model || modelSelect.value;
                byId("loadedModel").textContent = health.model || "Available";
                setSystemStatus("Model ready", "ready");
            } else {
                byId("loadedModel").textContent = "Unavailable";
                setSystemStatus("Checkpoint unavailable", "error");
            }
        } catch (error) {
            byId("loadedModel").textContent = "Server unavailable";
            setSystemStatus("Server unavailable", "error");
        }
    }

    function totalImages(counts = {}) {
        return Object.values(counts).reduce((sum, split) => sum + Object.values(split).reduce((a, b) => a + b, 0), 0);
    }

    function renderAudit(audit) {
        const passed = audit.audit_passed === true;
        const failed = audit.failed_datasets || ["busi", "oasbud", "breast"].filter((name) => {
            const item = audit[name] || {};
            return item.subject_ids_verifiable === false || Object.keys(item.cross_split_subjects || {}).length;
        });
        byId("evidenceStrip").classList.toggle("is-passed", passed);
        byId("auditSummary").textContent = passed
            ? "All included datasets passed identifier and cross-partition checks."
            : `Subject-level audit failed for ${failed.join(", ") || "the current cohort"}; metrics are engineering evidence only.`;
        byId("auditBadge").textContent = passed ? "Passed" : "Failed";
        byId("auditBadge").className = `status-badge ${passed ? "success" : "danger"}`;
        byId("auditDetail").textContent = passed
            ? "No cross-partition subject identifiers were found in the audited cohort."
            : "Known related images cross partitions, or source identifiers are insufficient to prove separation. Rebuild the cohort before reporting patient-independent performance.";

        const grid = byId("auditGrid");
        grid.replaceChildren();
        ["busi", "oasbud", "breast"].forEach((name) => {
            const item = audit[name] || {};
            const leaks = Object.keys(item.cross_split_subjects || {});
            const article = document.createElement("article");
            const heading = document.createElement("h4");
            const count = document.createElement("small");
            const detail = document.createElement("p");
            heading.textContent = name;
            count.textContent = `${totalImages(item.counts)} images in current folders`;
            detail.textContent = item.subject_ids_verifiable === false
                ? "Subject IDs not verifiable"
                : leaks.length ? `Cross-split subjects: ${leaks.join(", ")}` : "No cross-split IDs detected";
            article.append(heading, count, detail);
            grid.append(article);
        });
    }

    async function loadAudit() {
        try {
            const audit = await fetchJson("/api/data-audit");
            renderAudit(audit);
            return audit;
        } catch (error) {
            byId("auditSummary").textContent = "No audit file is available. Run scripts/audit_data.py before interpreting results.";
            byId("auditDetail").textContent = error.message;
            byId("auditBadge").textContent = "Unavailable";
            return null;
        }
    }

    async function loadEvidence() {
        try {
            const [payload] = await Promise.all([fetchJson("/api/benchmarks"), loadAudit()]);
            const metrics = payload.metrics;
            const summary = metrics.summary || {};
            byId("metricSamples").textContent = metrics.samples ?? "—";
            byId("metricAccuracy").textContent = Number.isFinite(summary.accuracy) ? `${(summary.accuracy * 100).toFixed(2)}%` : "—";
            byId("metricF1").textContent = Number.isFinite(summary.macro_f1) ? summary.macro_f1.toFixed(4) : "—";
            byId("metricAuc").textContent = Number.isFinite(summary.roc_auc) ? summary.roc_auc.toFixed(4) : "—";
            const generated = metrics.generated_at_utc ? new Date(metrics.generated_at_utc).toLocaleString() : "legacy run timestamp unavailable";
            byId("metricsProvenance").textContent = `${metrics.model_name || "Model"} · ${metrics.split || "test"} split · ${generated} · evidence status: ${metrics.evidence_status || "provisional"}.`;
            const list = byId("unavailableList");
            list.replaceChildren(...(payload.unavailable || []).map((item) => {
                const li = document.createElement("li");
                li.textContent = item;
                return li;
            }));
            [byId("confusionImage"), byId("rocImage")].forEach((image) => {
                const base = image.src.split("?")[0];
                image.src = `${base}?v=${Date.now()}`;
            });
            state.evidenceLoaded = true;
        } catch (error) {
            byId("metricsProvenance").textContent = error.message;
            toast("Evidence files could not be loaded.");
        }
    }
    byId("refreshEvidenceButton").addEventListener("click", loadEvidence);

    function inferenceForm(file, groundTruth = null) {
        const form = new FormData();
        form.append("file", file);
        form.append("use_clahe", String(claheToggle.checked));
        form.append("crop_strategy", cropSelect.value);
        form.append("model_name", modelSelect.value);
        if (groundTruth) form.append("ground_truth", groundTruth);
        return form;
    }

    function setSourcePreview(file, groundTruth) {
        if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
        state.imageObjectUrl = URL.createObjectURL(file);
        byId("sourceImage").src = state.imageObjectUrl;
        byId("sourceMeta").textContent = groundTruth ? `${file.name} · known label ${groundTruth.toLowerCase()}` : file.name;
    }

    function renderInference(result) {
        state.inference = result;
        byId("outputHeading").textContent = result.prediction?.toLowerCase() || "Unavailable";
        byId("confidenceValue").textContent = Number.isFinite(result.confidence) ? `${result.confidence.toFixed(2)}%` : "—";
        const benign = Number(result.probabilities?.benign || 0);
        const malignant = Number(result.probabilities?.malignant || 0);
        byId("benignValue").textContent = `${benign.toFixed(2)}%`;
        byId("malignantValue").textContent = `${malignant.toFixed(2)}%`;
        byId("benignBar").value = benign;
        byId("malignantBar").value = malignant;
        byId("benignBar").textContent = `${benign.toFixed(2)}%`;
        byId("malignantBar").textContent = `${malignant.toFixed(2)}%`;
        const morphology = result.morphology || {};
        byId("descriptorEcho").textContent = morphology.echogenicity || "Not estimated";
        byId("descriptorMargin").textContent = morphology.margin || "Not estimated";
        byId("descriptorPosterior").textContent = morphology.posterior_transmission || "Not estimated";
        byId("descriptorOrientation").textContent = morphology.orientation || "Not estimated";
        byId("resultRationale").textContent = result.research_report?.rationale || result.research_notice || "Research output only.";
        selectImageMode("gradcam");
    }

    function selectImageMode(mode) {
        if (!state.inference) return;
        const sources = {
            gradcam: [state.inference.gradcam_image, "Grad-CAM overlay"],
            processed: [state.inference.processed_image, "Processed model input"],
            clahe: [state.inference.clahe_image, "CLAHE preview"],
            raw: [state.inference.original_image, "Source image"],
        };
        const [src, label] = sources[mode] || sources.gradcam;
        byId("inspectionImage").src = src;
        byId("inspectionImage").alt = label;
        byId("inspectionLabel").textContent = label;
        document.querySelectorAll("[data-image-mode]").forEach((button) => button.classList.toggle("is-active", button.dataset.imageMode === mode));
    }
    document.querySelectorAll("[data-image-mode]").forEach((button) => button.addEventListener("click", () => selectImageMode(button.dataset.imageMode)));

    async function runSingle(file, groundTruth = null) {
        if (!file) return;
        state.currentFile = file;
        state.currentGroundTruth = groundTruth;
        state.inference = null;
        byId("consensusPanel").classList.add("is-hidden");
        byId("singleDropzone").classList.add("is-hidden");
        byId("inferenceLayout").classList.remove("is-hidden");
        byId("compareButton").disabled = true;
        setSourcePreview(file, groundTruth);
        setSystemStatus("Running inference…", "loading");
        try {
            const result = await fetchJson("/predict", { method: "POST", body: inferenceForm(file, groundTruth) });
            renderInference(result);
            byId("compareButton").disabled = false;
            byId("loadedModel").textContent = result.model_name || modelSelect.value;
            setSystemStatus("Model ready", "ready");
        } catch (error) {
            setSystemStatus("Inference failed", "error");
            showError(error.message);
        }
    }

    async function loadLabelledExample() {
        const cohort = await fetchJson("/api/dataset/cohort?dataset=busi&split=val&count=1");
        if (!cohort.length) throw new Error("No labelled example is available in the selected local folders.");
        const item = cohort[0];
        const response = await fetch(item.url);
        if (!response.ok) throw new Error("The example image could not be loaded.");
        const file = new File([await response.blob()], item.name, { type: response.headers.get("content-type") || "image/png" });
        return { file, groundTruth: item.ground_truth };
    }

    byId("singleFile").addEventListener("change", (event) => runSingle(event.target.files[0]));
    byId("singleDropzone").addEventListener("click", () => byId("singleFile").click());
    ["dragenter", "dragover"].forEach((name) => byId("singleDropzone").addEventListener(name, (event) => {
        event.preventDefault();
        byId("singleDropzone").classList.add("is-dragging");
    }));
    ["dragleave", "drop"].forEach((name) => byId("singleDropzone").addEventListener(name, (event) => {
        event.preventDefault();
        byId("singleDropzone").classList.remove("is-dragging");
    }));
    byId("singleDropzone").addEventListener("drop", (event) => runSingle(event.dataTransfer.files[0]));
    byId("loadSampleButton").addEventListener("click", async () => {
        try { const sample = await loadLabelledExample(); await runSingle(sample.file, sample.groundTruth); }
        catch (error) { showError(error.message); }
    });

    [modelSelect, cropSelect, claheToggle].forEach((control) => control.addEventListener("change", () => {
        if (state.currentFile) runSingle(state.currentFile, state.currentGroundTruth);
    }));

    byId("compareButton").addEventListener("click", async () => {
        if (!state.currentFile) return;
        const button = byId("compareButton");
        button.disabled = true;
        button.textContent = "Comparing…";
        try {
            const form = new FormData();
            form.append("file", state.currentFile);
            form.append("use_clahe", String(claheToggle.checked));
            form.append("crop_strategy", cropSelect.value);
            const data = await fetchJson("/predict/compare", { method: "POST", body: form });
            const grid = byId("comparisonGrid");
            grid.replaceChildren();
            Object.entries(data.models || {}).forEach(([name, result]) => {
                const article = document.createElement("article");
                const heading = document.createElement("h4");
                const detail = document.createElement("p");
                heading.textContent = name.replaceAll("_", " ");
                detail.textContent = result.error ? `Unavailable: ${result.error}` : `${result.prediction} · ${Number(result.confidence).toFixed(2)}%`;
                article.append(heading, detail);
                grid.append(article);
            });
            byId("consensusStatus").textContent = `${data.agreement_ratio || 0}% agreement`;
            byId("consensusPanel").classList.remove("is-hidden");
        } catch (error) { showError(error.message); }
        finally { button.disabled = false; button.textContent = "Compare available models"; }
    });

    byId("noiseIntensity").addEventListener("input", (event) => { byId("noiseIntensityValue").textContent = Number(event.target.value).toFixed(2); });
    byId("noiseSampleButton").addEventListener("click", async () => {
        try {
            const sample = state.currentFile ? { file: state.currentFile } : await loadLabelledExample();
            state.noiseFile = sample.file;
            byId("runNoiseButton").disabled = false;
            toast(`Noise input ready: ${sample.file.name}`);
        } catch (error) { showError(error.message); }
    });

    function renderNoiseStage(key, value) {
        const card = document.querySelector(`[data-stage="${key}"]`);
        card.querySelector("header strong").textContent = value.prediction || "—";
        const frame = card.querySelector(".stage-image");
        const image = new Image();
        image.src = value.image;
        image.alt = `${card.querySelector("header span").textContent} processed image`;
        frame.replaceChildren(image);
        const values = card.querySelectorAll("dd");
        values[0].textContent = Number.isFinite(value.confidence) ? `${Number(value.confidence).toFixed(2)}%` : "—";
        values[1].textContent = Number.isFinite(value.ssim) ? Number(value.ssim).toFixed(4) : "—";
    }

    byId("runNoiseButton").addEventListener("click", async () => {
        if (!state.noiseFile) return;
        const button = byId("runNoiseButton");
        button.disabled = true;
        button.textContent = "Running…";
        const form = new FormData();
        form.append("file", state.noiseFile);
        form.append("noise_type", byId("noiseType").value);
        form.append("intensity", byId("noiseIntensity").value);
        form.append("col3_strategy", byId("noiseComparator").value);
        form.append("use_clahe", String(claheToggle.checked));
        form.append("model_name", modelSelect.value);
        form.append("seed", byId("noiseSeed").value);
        try {
            const data = await fetchJson("/api/noise/simulate", { method: "POST", body: form });
            ["stage1_clean", "stage2_noisy", "stage3_direct", "stage4_roi"].forEach((key) => renderNoiseStage(key, data[key]));
            byId("noiseExplanation").textContent = `${data.explanation} Seed ${data.seed}; model ${data.model_name}. Interpret only as a controlled single-image demonstration.`;
        } catch (error) { showError(error.message); }
        finally { button.disabled = false; button.textContent = "Run experiment"; }
    });

    async function runBatch(files, labels = new Map()) {
        if (!files.length) return;
        state.batchLabels = labels;
        const form = new FormData();
        files.forEach((file) => form.append("files", file));
        form.append("use_clahe", String(claheToggle.checked));
        form.append("crop_strategy", cropSelect.value);
        form.append("model_name", modelSelect.value);
        setSystemStatus("Processing batch…", "loading");
        try {
            const data = await fetchJson("/predict/batch", { method: "POST", body: form });
            state.batchResults = (data.results || []).map((result) => ({ ...result, ground_truth: labels.get(result.filename) || "—" }));
            byId("batchProcessed").textContent = data.cohort_summary?.processed_scans ?? 0;
            byId("batchBenign").textContent = data.cohort_summary?.benign_count ?? 0;
            byId("batchMalignant").textContent = data.cohort_summary?.malignant_count ?? 0;
            const mean = data.cohort_summary?.average_confidence;
            byId("batchConfidence").textContent = Number.isFinite(mean) ? `${Number(mean).toFixed(2)}%` : "—";
            byId("exportBatchButton").disabled = !state.batchResults.length;
            renderBatch();
            setSystemStatus("Model ready", "ready");
        } catch (error) {
            setSystemStatus("Batch failed", "error");
            showError(error.message);
        }
    }

    function renderBatch() {
        const filter = byId("batchFilter").value;
        const rows = state.batchResults.filter((result) => {
            if (filter === "all") return true;
            if (filter === "error") return result.status !== "success";
            return result.prediction?.toLowerCase() === filter;
        });
        const body = byId("batchBody");
        body.replaceChildren();
        if (!rows.length) {
            const row = body.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 6;
            cell.className = "empty-cell";
            cell.textContent = "No results match this filter.";
            return;
        }
        rows.forEach((result) => {
            const row = body.insertRow();
            const values = result.status === "success"
                ? [result.filename, result.ground_truth, result.prediction, `${Number(result.confidence).toFixed(2)}%`, (result.output_band || "—").replaceAll("_", " "), "research output"]
                : [result.filename, result.ground_truth, "—", "—", "—", result.error || result.status];
            values.forEach((value, index) => {
                const cell = row.insertCell();
                cell.textContent = value;
                if (index === 5) cell.className = "table-status";
            });
        });
    }

    byId("batchFiles").addEventListener("change", (event) => runBatch([...event.target.files]));
    byId("loadCohortButton").addEventListener("click", async () => {
        const button = byId("loadCohortButton");
        button.disabled = true;
        button.textContent = "Loading…";
        try {
            const cohort = await fetchJson("/api/dataset/cohort?dataset=busi&split=test&count=12");
            const files = await Promise.all(cohort.map(async (item) => {
                const response = await fetch(item.url);
                if (!response.ok) throw new Error(`Could not load ${item.name}.`);
                return new File([await response.blob()], `${item.ground_truth}_${item.name}`, { type: response.headers.get("content-type") || "image/png" });
            }));
            const labels = new Map(cohort.map((item) => [`${item.ground_truth}_${item.name}`, item.ground_truth]));
            await runBatch(files, labels);
        } catch (error) { showError(error.message); }
        finally { button.disabled = false; button.textContent = "Load labelled cohort"; }
    });
    byId("batchFilter").addEventListener("change", renderBatch);

    function csvCell(value) { return `"${String(value ?? "").replaceAll('"', '""')}"`; }
    byId("exportBatchButton").addEventListener("click", () => {
        const header = ["filename", "known_label", "model_output", "softmax_score_percent", "output_band", "status"];
        const lines = [header, ...state.batchResults.map((r) => [r.filename, r.ground_truth, r.prediction || "", r.confidence ?? "", r.output_band || "", r.status])];
        const blob = new Blob([lines.map((row) => row.map(csvCell).join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "research_batch_outputs.csv";
        link.click();
        URL.revokeObjectURL(link.href);
    });

    async function loadDataset() {
        const grid = byId("datasetGrid");
        grid.innerHTML = '<p class="empty-state">Loading local images…</p>';
        try {
            const dataset = byId("datasetSelect").value;
            const split = byId("splitSelect").value;
            const files = await fetchJson(`/api/dataset/files?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}`);
            grid.replaceChildren();
            if (!files.length) {
                byId("datasetSummary").textContent = "No images found.";
                grid.innerHTML = '<p class="empty-state">No images were found in this folder.</p>';
                return;
            }
            const preview = files.slice(0, 24);
            byId("datasetSummary").textContent = `Showing ${preview.length} of ${files.length} images. Counts reflect folder contents, not independent subjects.`;
            preview.forEach((file) => {
                const article = document.createElement("article");
                article.className = "dataset-item";
                const image = new Image();
                image.loading = "lazy";
                image.src = file.url;
                image.alt = `${file.class} folder example ${file.name}`;
                const meta = document.createElement("div");
                const name = document.createElement("strong");
                const label = document.createElement("span");
                name.textContent = file.name;
                label.textContent = `${file.class} folder label`;
                meta.append(name, label);
                article.append(image, meta);
                grid.append(article);
            });
        } catch (error) {
            byId("datasetSummary").textContent = "Dataset preview unavailable.";
            grid.innerHTML = `<p class="empty-state">${error.message}</p>`;
        }
    }
    byId("datasetSelect").addEventListener("change", loadDataset);
    byId("splitSelect").addEventListener("change", loadDataset);

    async function loadDocument(filename) {
        const viewer = byId("documentViewer");
        viewer.innerHTML = "<p>Loading document…</p>";
        try {
            const payload = await fetchJson(`/api/code/${encodeURIComponent(filename)}`);
            if (window.marked) viewer.innerHTML = window.marked.parse(payload.code);
            else {
                const pre = document.createElement("pre");
                pre.textContent = payload.code;
                viewer.replaceChildren(pre);
            }
            viewer.dataset.loaded = filename;
            document.querySelectorAll("[data-document]").forEach((button) => button.classList.toggle("is-active", button.dataset.document === filename));
            if (window.renderMathInElement) window.renderMathInElement(viewer, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }] });
        } catch (error) { viewer.textContent = error.message; }
    }
    document.querySelectorAll("[data-document]").forEach((button) => button.addEventListener("click", () => loadDocument(button.dataset.document)));
    byId("documentViewer").addEventListener("click", (event) => {
        const link = event.target.closest("a");
        if (!link) return;
        const filename = link.getAttribute("href")?.split("/").pop();
        if (!filename || !document.querySelector(`[data-document="${CSS.escape(filename)}"]`)) return;
        event.preventDefault();
        loadDocument(filename);
    });

    checkHealth();
    loadAudit();
    switchView(location.hash.slice(1) || "workstation", false);
});
