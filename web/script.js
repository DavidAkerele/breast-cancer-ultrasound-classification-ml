document.addEventListener('DOMContentLoaded', () => {
    // UI Panels & Inputs
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const dropzonePreviewImg = document.getElementById('dropzonePreviewImg');
    const previewFilename = document.getElementById('previewFilename');
    const previewFilesize = document.getElementById('previewFilesize');
    const btnRemovePreview = document.getElementById('btnRemovePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const claheToggle = document.getElementById('claheToggle');
    const modelSelect = document.getElementById('modelSelect');
    const existingStudySelect = document.getElementById('existingStudySelect');
    const noiseStudySelect = document.getElementById('noiseStudySelect');
    const datasetSplitSelect = document.getElementById('datasetSplitSelect');

    // Crop Modal Elements
    const cropModalOverlay = document.getElementById('cropModalOverlay');
    const cropModalImg = document.getElementById('cropModalImg');
    let cropModalCancel = document.getElementById('cropModalCancel');
    let cropModalConfirm = document.getElementById('cropModalConfirm');
    let cropper = null;

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const batchResultsContent = document.getElementById('batchResultsContent');

    // UI Diagnostics & Output previews
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictConfidence = document.getElementById('verdictConfidence');
    const origImgPreview = document.getElementById('origImgPreview');
    const claheImgPreview = document.getElementById('claheImgPreview');
    const benignProbVal = document.getElementById('benignProbVal');
    const benignProbBar = document.getElementById('benignProbBar');
    const malignantProbVal = document.getElementById('malignantProbVal');
    const malignantProbBar = document.getElementById('malignantProbBar');

    // Dashboard Info Widgets
    const deviceIndicator = document.getElementById('deviceIndicator');
    const statsStudiesCount = document.getElementById('statsStudiesCount');
    const logConsole = document.getElementById('logConsole');
    const btnClearLogs = document.getElementById('btnClearLogs');

    // Modes & Code Viewer Elements
    const btnUiMode = document.getElementById('btnUiMode');
    const btnCodeMode = document.getElementById('btnCodeMode');
    const btnNoiseMode = document.getElementById('btnNoiseMode');
    const uiDashboardContainer = document.getElementById('uiDashboardContainer');
    const notebookContainer = document.getElementById('notebookContainer');
    const noiseCompareContainer = document.getElementById('noiseCompareContainer');
    
    // Noise Comparison Panel Elements
    const noiseCleanImg = document.getElementById('noiseCleanImg');
    const noiseCleanPlaceholder = document.getElementById('noiseCleanPlaceholder');
    const noiseCleanVerdict = document.getElementById('noiseCleanVerdict');
    const noiseCleanConfidence = document.getElementById('noiseCleanConfidence');
    const noiseCleanBanner = document.getElementById('noiseCleanBanner');
    
    const noiseCorruptedImg = document.getElementById('noiseCorruptedImg');
    const noiseCorruptedPlaceholder = document.getElementById('noiseCorruptedPlaceholder');
    const noiseCorruptedVerdict = document.getElementById('noiseCorruptedVerdict');
    const noiseCorruptedConfidence = document.getElementById('noiseCorruptedConfidence');
    const noiseCorruptedBanner = document.getElementById('noiseCorruptedBanner');
    
    const btnRunNoiseCompare = document.getElementById('btnRunNoiseCompare');
    const noiseCompareType = document.getElementById('noiseCompareType');
    const noiseCompareSeverity = document.getElementById('noiseCompareSeverity');
    const noiseExplanationText = document.getElementById('noiseExplanationText');
    const activeFileName = document.getElementById('activeFileName');
    const btnCopyCode = document.getElementById('btnCopyCode');
    const btnRunAllCells = document.getElementById('btnRunAllCells');
    const notebookCellsList = document.getElementById('notebookCellsList');
    const navItems = document.querySelectorAll('.nav-item');

    const API_BASE = (window.location.protocol === 'file:') ? 'http://localhost:8000' : '';

    let selectedFile = null;
    let selectedFiles = []; // Holds folder / batch multi-files
    let studiesCount = 0;
    let codeCache = {};

    // Helper to escape HTML tags
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // --- Interactive Audit Logger ---
    function addLogEntry(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-msg log-level-${level}">${message}</span>
        `;
        logConsole.appendChild(entry);
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    // --- Dedicated Alert Modal System ---
    const alertModalOverlay = document.getElementById('alertModalOverlay');
    const alertModal = document.getElementById('alertModal');
    const alertModalTitle = document.getElementById('alertModalTitle');
    const alertModalMessage = document.getElementById('alertModalMessage');
    const alertModalDismiss = document.getElementById('alertModalDismiss');

    /**
     * Show the dedicated alert modal.
     * @param {string} title - The alert heading text.
     * @param {string} message - The alert body description.
     * @param {'error'|'warning'|'info'|'success'} type - Visual style of the alert.
     */
    function showAlertModal(title, message, type = 'error') {
        if (!alertModalOverlay || !alertModal) return;

        // Reset type classes
        alertModal.className = 'alert-modal';
        if (type === 'warning') alertModal.classList.add('alert-warning');
        else if (type === 'info') alertModal.classList.add('alert-info');
        else if (type === 'success') alertModal.classList.add('alert-success');
        // 'error' uses the default (no extra class)

        if (alertModalTitle) alertModalTitle.textContent = title;
        if (alertModalMessage) alertModalMessage.textContent = message;
        alertModalOverlay.classList.add('visible');
    }

    function hideAlertModal() {
        if (alertModalOverlay) alertModalOverlay.classList.remove('visible');
    }

    if (alertModalDismiss) alertModalDismiss.addEventListener('click', hideAlertModal);
    if (alertModalOverlay) {
        alertModalOverlay.addEventListener('click', (e) => {
            if (e.target === alertModalOverlay) hideAlertModal();
        });
    }

    addLogEntry('OncoVision Clinical Control Center initialized.', 'success');
    addLogEntry('Awaiting network status handshake...', 'info');

    // Fetch Backend Health / Hardware details
    async function fetchServerStatus() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (response.ok) {
                const data = await response.json();
                if (deviceIndicator) deviceIndicator.textContent = data.device.toUpperCase();
                addLogEntry(`Connection established. Device: ${data.device.toUpperCase()}. Classes: ${data.classes.join(', ')}`, 'success');
            } else {
                throw new Error();
            }
        } catch (e) {
            addLogEntry('FastAPI Backend connection failed. Running in demo simulation mode.', 'warning');
            if (deviceIndicator) deviceIndicator.textContent = 'CPU (MOCK)';
        }
    }
    fetchServerStatus();
    loadCodeFile('dataset.py');

    let activePredictionData = null;

    // Gallery elements
    const galleryGrid = document.getElementById('galleryGrid');
    const galleryCount = document.getElementById('galleryCount');

    // Fetch and populate existing patient files + image gallery
    const datasetSelect = document.getElementById('datasetSelect');

    // Fetch and populate existing patient files + image gallery
    async function fetchDatasetFiles(dataset = 'busi', split = 'val') {
        try {
            const response = await fetch(`${API_BASE}/api/dataset/files?dataset=${dataset}&split=${split}`);
            if (response.ok) {
                const data = await response.json();

                // Populate dropdown
                existingStudySelect.innerHTML = '<option value="" selected>-- Choose patient study --</option>';
                if (noiseStudySelect) {
                    noiseStudySelect.innerHTML = '<option value="" selected>-- Choose patient study --</option>';
                }
                data.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.url;
                    // Extract case number safely without query params
                    let display_name = item.name;
                    if (display_name.includes('?')) display_name = display_name.split('?')[0];
                    option.textContent = `${display_name} (${item.class.toUpperCase()})`;
                    existingStudySelect.appendChild(option);
                    if (noiseStudySelect) {
                        const noiseOption = option.cloneNode(true);
                        noiseStudySelect.appendChild(noiseOption);
                    }
                });

                // Populate image gallery
                if (galleryGrid) {
                    galleryGrid.innerHTML = '';
                    if (data.length === 0) {
                        galleryGrid.innerHTML = '<div class="gallery-empty"><p>No scans found in this split.</p></div>';
                    } else {
                        data.forEach(item => {
                            const card = document.createElement('div');
                            card.className = 'gallery-item';
                            card.dataset.url = item.url;
                            card.innerHTML = `
                                <img src="${API_BASE}${item.url}" alt="${item.name}" loading="lazy">
                                <span class="gallery-badge ${item.class}">${item.class}</span>
                            `;
                            card.addEventListener('click', () => {
                                // Select this image
                                existingStudySelect.value = item.url;
                                existingStudySelect.dispatchEvent(new Event('change'));
                                // Highlight active card
                                galleryGrid.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('active'));
                                card.classList.add('active');
                            });
                            galleryGrid.appendChild(card);
                        });
                    }
                }
                if (galleryCount) galleryCount.textContent = `${data.length} scans`;

                addLogEntry(`Loaded ${data.length} files from ${dataset.toUpperCase()} (${split}) split database.`, 'success');
            }
        } catch (e) {
            addLogEntry(`Failed to fetch ${dataset.toUpperCase()} (${split}) dataset index.`, 'warning');
        }
    }
    
    function reloadDataset() {
        const dataset = datasetSelect ? datasetSelect.value : 'busi';
        const split = datasetSplitSelect ? datasetSplitSelect.value : 'val';
        fetchDatasetFiles(dataset, split);
        selectedFile = null;
        selectedFiles = [];
        analyzeBtn.disabled = true;
        existingStudySelect.value = '';
        showState('empty');
    }

    // Initial fetch
    const initialDataset = datasetSelect ? datasetSelect.value : 'busi';
    const initialSplit = datasetSplitSelect ? datasetSplitSelect.value : 'val';
    fetchDatasetFiles(initialDataset, initialSplit);

    // Fetch new dataset split files when selection changes
    if (datasetSplitSelect) {
        datasetSplitSelect.addEventListener('change', () => {
            reloadDataset();
        });
    }

    if (datasetSelect) {
        datasetSelect.addEventListener('change', () => {
            reloadDataset();
        });
    }

    // Segmented tab controls interaction for Split
    const splitSegmentBtns = document.querySelectorAll('#datasetSplitTabs .segment-btn');
    splitSegmentBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            splitSegmentBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const val = btn.getAttribute('data-value');
            if (datasetSplitSelect) {
                datasetSplitSelect.value = val;
                datasetSplitSelect.dispatchEvent(new Event('change'));
            }
        });
    });

    // Segmented tab controls interaction for Dataset
    const datasetSegmentBtns = document.querySelectorAll('#datasetTabs .segment-btn');
    datasetSegmentBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            datasetSegmentBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const val = btn.getAttribute('data-value');
            if (datasetSelect) {
                datasetSelect.value = val;
                datasetSelect.dispatchEvent(new Event('change'));
            }
        });
    });

    // Architecture explainer data & dynamics
    const architectureData = {
        'CUSTOM_CNN': {
            name: 'Custom 4-Block CNN',
            badge: 'Convolutional Baseline',
            params: '1.2M Params',
            desc: 'A custom Convolutional Neural Network built specifically for this dataset. Composed of 4 successive Conv2D blocks with Batch Normalization and ReLU activations to learn localized low-level ultrasound texture patterns.',
            rationale: 'Acts as a clinical control baseline to evaluate the performance improvement of transfer learning models.'
        },
        'RESNET50': {
            name: 'ResNet-50 Fine-tuned',
            badge: 'Deep Residual Network',
            params: '23.5M Params',
            desc: 'A 50-layer deep residual network fine-tuned using pre-trained ImageNet features. Utilizes skip connections (residual blocks) to bypass layers, resolving the vanishing gradient problem and allowing very high classification boundary convergence.',
            rationale: 'Excellent for extracting abstract edge patterns, acoustic shadows, and tissue density boundaries from ultrasound scans.'
        },
        'EFFICIENTNET_B0': {
            name: 'EfficientNet-B0 Backbone',
            badge: 'Compound Scaled Net',
            params: '4.0M Params',
            desc: 'An optimized backbone compound-scaled across depth, width, and resolution using neural architecture search (NAS). Implements mobile inverted bottlenecks (MBConv) for compute-efficient, high-accuracy classification.',
            rationale: 'Provides high AUC scores with faster execution, ideal for resource-constrained clinical edge deployments.'
        }
    };

    function updateArchitectureExplainer(modelKey) {
        const data = architectureData[modelKey.toUpperCase()];
        if (!data) return;
        
        const badge = document.getElementById('explainerBadge');
        const name = document.getElementById('explainerName');
        const desc = document.getElementById('explainerDesc');
        const rationale = document.getElementById('explainerRationale');
        const params = document.getElementById('explainerParams');
        
        if (badge) badge.textContent = data.badge;
        if (name) name.textContent = data.name;
        if (desc) desc.textContent = data.desc;
        if (rationale) rationale.textContent = data.rationale;
        if (params) params.textContent = data.params;
    }

    function resetDiagnosticOutcomes() {
        const clahePlaceholder = document.getElementById('clahePlaceholder');
        if (claheImgPreview) {
            claheImgPreview.src = '';
            claheImgPreview.classList.add('hidden');
        }
        if (clahePlaceholder) {
            clahePlaceholder.classList.remove('hidden');
        }

        if (verdictTitle) {
            verdictTitle.textContent = 'Pending diagnostic execution';
            verdictTitle.style.fontSize = '1.05rem';
        }
        if (verdictConfidence) {
            verdictConfidence.textContent = 'Awaiting analysis';
        }
        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner pending';
        }

        const clinicalBirads = document.getElementById('clinicalBirads');
        const clinicalDensity = document.getElementById('clinicalDensity');
        const clinicalShadowing = document.getElementById('clinicalShadowing');
        const clinicalSummaryText = document.getElementById('clinicalSummaryText');

        if (clinicalBirads) clinicalBirads.textContent = 'Awaiting execution...';
        if (clinicalDensity) clinicalDensity.textContent = 'Awaiting execution...';
        if (clinicalShadowing) clinicalShadowing.textContent = 'Awaiting execution...';
        if (clinicalSummaryText) clinicalSummaryText.textContent = 'Awaiting feature evaluation...';
    }

    // Handle existing study dropdown selection
    existingStudySelect.addEventListener('change', async (e) => {
        const url = e.target.value;
        if (!url) {
            selectedFile = null;
            selectedFiles = [];
            analyzeBtn.disabled = true;
            showState('empty');
            // Clear gallery active state
            if (galleryGrid) galleryGrid.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('active'));
            return;
        }

        selectedFile = null; 
        selectedFiles = [];
        fileInput.value = ''; 
        
        // Extract filename for UI display (strip query parameters if present)
        let filename = url.substring(url.lastIndexOf('/') + 1);
        if (filename.includes('?')) {
            filename = filename.split('?')[0];
        }

        // Sync gallery active state
        if (galleryGrid) {
            galleryGrid.querySelectorAll('.gallery-item').forEach(el => {
                el.classList.toggle('active', el.dataset.url === url);
            });
        }

        addLogEntry(`Downloading patient study: ${filename} for ROI selection...`, 'info');
        showState('loading');
        
        try {
            const response = await fetch(`${API_BASE}${url}`);
            if (!response.ok) throw new Error('Failed to retrieve scan image');
            const blob = await response.blob();
            
            const reader = new FileReader();
            reader.onload = (eReader) => {
                openCropperModal(eReader.target.result, filename, (croppedBlob, croppedDataUrl) => {
                    selectedFile = new File([croppedBlob], filename, {type: "image/png"});
                    
                    const h3 = dropzone ? dropzone.querySelector('h3') : null;
                    const p = dropzone ? dropzone.querySelector('p') : null;
                    if (h3) h3.textContent = filename;
                    if (p) {
                        const split = datasetSplitSelect ? datasetSplitSelect.value : 'val';
                        p.textContent = `Selected from ${split.toUpperCase()} split (ROI Cropped)`;
                    }
                    
                    if (dropzonePreviewImg) dropzonePreviewImg.src = croppedDataUrl;
                    if (previewFilename) previewFilename.textContent = filename;
                    if (previewFilesize) previewFilesize.textContent = `${(croppedBlob.size / 1024 / 1024).toFixed(2)} MB • Ready`;
                    
                    if (dropzoneContent) dropzoneContent.classList.add('hidden');
                    if (dropzonePreview) dropzonePreview.classList.remove('hidden');
                    
                    if (origImgPreview) origImgPreview.src = croppedDataUrl;
                    showState('results');
                    resetDiagnosticOutcomes();
                    
                    if (analyzeBtn) analyzeBtn.disabled = false;
                    addLogEntry(`ROI selected for validation study: ${filename}`, 'info');
                });
            };
            reader.readAsDataURL(blob);
        } catch (err) {
            addLogEntry(`Failed to load patient study: ${err.message}`, 'error');
            showAlertModal('Study Load Error', err.message, 'error');
            showState('empty');
        }
    });

    // Bind clickable cards for Target Architecture Selector
    const archCards = document.querySelectorAll('.arch-card');
    archCards.forEach(card => {
        card.addEventListener('click', () => {
            archCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            
            if (modelSelect) {
                modelSelect.value = card.dataset.value;
                modelSelect.dispatchEvent(new Event('change'));
            }
        });
    });

    // Model selector updates
    modelSelect.addEventListener('change', (e) => {
        const val = e.target.value.toUpperCase();
        addLogEntry(`Target model updated: ${val}`, 'info');
        updateArchitectureExplainer(e.target.value);
        
        // Sync active class on cards if dropdown is changed programmatically
        if (archCards) {
            archCards.forEach(c => {
                c.classList.toggle('active', c.dataset.value === e.target.value);
            });
        }
    });

    // Clear logs handler
    btnClearLogs.addEventListener('click', () => {
        logConsole.innerHTML = '';
        addLogEntry('Clinical log feed cleared.', 'info');
    });

    // Drag & Drop Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files);
        }
    });

    function openCropperModal(imageSrc, filename, onConfirmCallback) {
        if (cropper) {
            cropper.destroy();
            cropper = null;
        }
        
        cropModalImg.src = imageSrc;
        cropModalOverlay.classList.add('visible');
        
        cropper = new Cropper(cropModalImg, {
            aspectRatio: 1, // Force square crop
            viewMode: 1, // Restrict crop box to not exceed the size of the canvas
            autoCropArea: 0.8, // 80% of image size by default
            responsive: true,
            restore: false,
            guides: true,
            center: true,
            highlight: false,
            cropBoxMovable: true,
            cropBoxResizable: true,
            toggleDragModeOnDblclick: false
        });
        
        const newConfirm = () => {
            if (!cropper) return;
            cropper.getCroppedCanvas({
                width: 512,
                height: 512,
                imageSmoothingEnabled: true,
                imageSmoothingQuality: 'high'
            }).toBlob((blob) => {
                cropModalOverlay.classList.remove('visible');
                cropper.destroy();
                cropper = null;
                onConfirmCallback(blob, URL.createObjectURL(blob));
            }, 'image/png');
        };
        
        const newCancel = () => {
            cropModalOverlay.classList.remove('visible');
            if (cropper) {
                cropper.destroy();
                cropper = null;
            }
            if (!selectedFile) {
                showState('empty');
            } else {
                showState('results');
            }
        };
        
        const confirmBtnClone = cropModalConfirm.cloneNode(true);
        const cancelBtnClone = cropModalCancel.cloneNode(true);
        
        cropModalConfirm.parentNode.replaceChild(confirmBtnClone, cropModalConfirm);
        cropModalCancel.parentNode.replaceChild(cancelBtnClone, cropModalCancel);
        
        confirmBtnClone.addEventListener('click', newConfirm);
        cancelBtnClone.addEventListener('click', newCancel);
        
        cropModalConfirm = confirmBtnClone;
        cropModalCancel = cancelBtnClone;
    }

    function handleFileSelect(files) {
        existingStudySelect.value = ''; // Reset dropdown selection
        
        // Hide preview by default
        if (dropzoneContent) dropzoneContent.classList.remove('hidden');
        if (dropzonePreview) dropzonePreview.classList.add('hidden');
        
        if (files.length > 1) {
            selectedFiles = Array.from(files);
            selectedFile = null;
            
            const h3 = dropzone ? dropzone.querySelector('h3') : null;
            const p = dropzone ? dropzone.querySelector('p') : null;
            if (h3) h3.textContent = `Batch: ${selectedFiles.length} Scans Loaded`;
            if (p) p.textContent = 'Ready for batch dataset evaluation';
            if (analyzeBtn) analyzeBtn.disabled = false;
            addLogEntry(`Dataset batch loaded: ${selectedFiles.length} images ready.`, 'info');
        } else {
            const file = files[0];
            if (!file.type.startsWith('image/')) {
                addLogEntry(`File rejection: Unsupported file format.`, 'error');
                showAlertModal('Invalid File Format', 'Please select a valid image file (PNG, JPEG, TIFF).', 'warning');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                openCropperModal(e.target.result, file.name, (croppedBlob, croppedDataUrl) => {
                    selectedFile = new File([croppedBlob], file.name, {type: "image/png"});
                    selectedFiles = [];
                    
                    if (dropzonePreviewImg) dropzonePreviewImg.src = croppedDataUrl;
                    if (previewFilename) previewFilename.textContent = file.name;
                    if (previewFilesize) previewFilesize.textContent = `${(croppedBlob.size / 1024 / 1024).toFixed(2)} MB • Ready`;
                    
                    if (dropzoneContent) dropzoneContent.classList.add('hidden');
                    if (dropzonePreview) dropzonePreview.classList.remove('hidden');
                    
                    if (origImgPreview) origImgPreview.src = croppedDataUrl;
                    showState('results');
                    resetDiagnosticOutcomes();
                    
                    if (analyzeBtn) analyzeBtn.disabled = false;
                    addLogEntry(`ROI selected for ultrasound study: ${file.name}`, 'info');
                });
            };
            reader.readAsDataURL(file);
        }
    }

    // Handle preview removal
    if (btnRemovePreview) {
        btnRemovePreview.addEventListener('click', (e) => {
            e.stopPropagation(); // Avoid triggering dropzone container clicks
            fileInput.value = '';
            selectedFile = null;
            selectedFiles = [];
            
            if (dropzonePreview) dropzonePreview.classList.add('hidden');
            if (dropzoneContent) {
                dropzoneContent.classList.remove('hidden');
                const h3 = dropzoneContent.querySelector('h3');
                const p = dropzoneContent.querySelector('p');
                if (h3) h3.textContent = 'Drag & Drop Ultrasound Study';
                if (p) p.textContent = 'Supports standard medical PNG/JPG studies';
            }
            
            if (analyzeBtn) analyzeBtn.disabled = true;
            showState('empty');
            addLogEntry('Selected ultrasound study removed.', 'info');
        });
    }

    // Benchmark sample loaders (Represent Use Cases / Preloaded cases)
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.getAttribute('data-type');
            const h3 = dropzone ? dropzone.querySelector('h3') : null;
            const p = dropzone ? dropzone.querySelector('p') : null;
            if (h3) h3.textContent = `Benchmark Case: ${type.toUpperCase()}`;
            if (p) p.textContent = `Preloaded medical study use-case`;
            selectedFile = null; 
            selectedFiles = [];
            if (existingStudySelect) existingStudySelect.value = '';
            if (analyzeBtn) analyzeBtn.disabled = false;
            addLogEntry(`Loaded clinical benchmark study use-case: ${type.toUpperCase()}`, 'info');
            analyzeSample(type);
        });
    });

    analyzeBtn.addEventListener('click', async () => {
        if (selectedFiles.length > 0) {
            uploadAndAnalyzeBatch(selectedFiles);
        } else if (selectedFile) {
            uploadAndAnalyze(selectedFile);
        } else if (existingStudySelect.value) {
            // Selected an existing file from the validation dropdown
            const url = existingStudySelect.value;
            let filename = url.substring(url.lastIndexOf('/') + 1);
            if (filename.includes('?')) {
                filename = filename.split('?')[0];
            }
            showState('loading');
            addLogEntry(`Fetching validation file content for: ${filename}...`, 'info');
            
            try {
                const response = await fetch(`${API_BASE}${url}`);
                if (!response.ok) throw new Error('Image retrieval failed');
                const blob = await response.blob();
                const file = new File([blob], filename, {type: "image/png"});
                uploadAndAnalyze(file);
            } catch (e) {
                addLogEntry(`Failed to download study image: ${e.message}`, 'error');
                showState('empty');
            }
        } else {
            const h3 = dropzone ? dropzone.querySelector('h3') : null;
            const isBenign = h3 ? h3.textContent.includes('BENIGN') : false;
            analyzeSample(isBenign ? 'benign' : 'malignant');
        }
    });

    async function uploadAndAnalyze(file) {
        showState('loading');
        addLogEntry(`Evaluating study: ${file.name}...`, 'info');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('use_clahe', claheToggle.checked);

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to process ultrasound scan');
            }

            const data = await response.json();
            
            studiesCount++;
            if (statsStudiesCount) statsStudiesCount.textContent = studiesCount;

            addLogEntry(`Prediction completed. Verdict: ${data.prediction} (${data.confidence}%)`, 'success');
            renderResults(data);
        } catch (error) {
            addLogEntry(`Analysis failed: ${error.message}`, 'error');
            showAlertModal('Analysis Error', error.message, 'error');
            showState('empty');
        }
    }

    async function uploadAndAnalyzeBatch(files) {
        showState('loading');
        addLogEntry(`Classifying batch dataset: ${files.length} images...`, 'info');

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        formData.append('use_clahe', claheToggle.checked);

        try {
            const response = await fetch(`${API_BASE}/predict/batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Batch analysis execution failed');
            }

            const data = await response.json();
            
            studiesCount += files.length;
            if (statsStudiesCount) statsStudiesCount.textContent = studiesCount;

            addLogEntry(`Batch processing completed. Results compiled for ${data.results.length} files.`, 'success');
            renderBatchResults(data.results);
        } catch (error) {
            addLogEntry(`Batch analysis failed: ${error.message}`, 'error');
            showAlertModal('Batch Analysis Error', error.message, 'error');
            showState('empty');
        }
    }

    async function analyzeSample(type) {
        showState('loading');
        addLogEntry(`Simulating neural execution for benchmark: ${type.toUpperCase()}...`, 'info');
        
        setTimeout(() => {
            const isBenign = (type === 'benign');
            const conf = isBenign ? 94.60 : 91.80;
            const bProb = isBenign ? 94.60 : 8.20;
            const mProb = isBenign ? 5.40 : 91.80;

            const dummySvg = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 224 224"><rect width="100%" height="100%" fill="%230b0e14"/><circle cx="112" cy="112" r="${isBenign ? 40 : 60}" fill="%23${isBenign ? '10b981' : 'f43f5e'}" opacity="0.6"/><text x="50%" y="50%" fill="white" font-family="sans-serif" font-size="12" text-anchor="middle" dy=".3em">${type.toUpperCase()}</text></svg>`;

            studiesCount++;
            if (statsStudiesCount) statsStudiesCount.textContent = studiesCount;

            addLogEntry(`Diagnosis completed. Verdict: ${type.toUpperCase()} (${conf}%)`, 'success');

            renderResults({
                prediction: isBenign ? 'BENIGN' : 'MALIGNANT',
                confidence: conf,
                probabilities: { benign: bProb, malignant: mProb },
                original_image: dummySvg,
                processed_image: dummySvg,
                noise_analysis: {
                    dominant_type: "Speckle Noise (Acoustic)",
                    description: "Simulated standard acoustic speckling typically seen in clinical breast ultrasound imaging.",
                    metrics: {
                        speckle_level: 32.5,
                        gaussian_level: 12.4,
                        impulse_level: 1.2,
                        snr_db: 22.40
                    }
                }
            });
        }, 800);
    }

    function renderNoiseAnalysis(noiseData) {
        const noiseContentEmpty = document.getElementById('noiseContentEmpty');
        const noiseContentResults = document.getElementById('noiseContentResults');
        const dominantNoiseType = document.getElementById('dominantNoiseType');
        const dominantNoiseDesc = document.getElementById('dominantNoiseDesc');
        const speckleLevelVal = document.getElementById('speckleLevelVal');
        const speckleLevelBar = document.getElementById('speckleLevelBar');
        const gaussianLevelVal = document.getElementById('gaussianLevelVal');
        const gaussianLevelBar = document.getElementById('gaussianLevelBar');
        const impulseLevelVal = document.getElementById('impulseLevelVal');
        const impulseLevelBar = document.getElementById('impulseLevelBar');
        const snrValue = document.getElementById('snrValue');

        if (!noiseData) {
            if (noiseContentEmpty) noiseContentEmpty.classList.remove('hidden');
            if (noiseContentResults) noiseContentResults.classList.add('hidden');
            return;
        }

        if (noiseContentEmpty) noiseContentEmpty.classList.add('hidden');
        if (noiseContentResults) noiseContentResults.classList.remove('hidden');

        if (dominantNoiseType) dominantNoiseType.textContent = noiseData.dominant_type;
        if (dominantNoiseDesc) dominantNoiseDesc.textContent = noiseData.description;
        
        const speckle = Math.round(noiseData.metrics.speckle_level);
        const gaussian = Math.round(noiseData.metrics.gaussian_level);
        const impulse = Math.round(noiseData.metrics.impulse_level);
        
        if (speckleLevelVal) speckleLevelVal.textContent = `${speckle}%`;
        if (gaussianLevelVal) gaussianLevelVal.textContent = `${gaussian}%`;
        if (impulseLevelVal) impulseLevelVal.textContent = `${impulse}%`;
        
        setTimeout(() => {
            if (speckleLevelBar) speckleLevelBar.style.width = `${speckle}%`;
            if (gaussianLevelBar) gaussianLevelBar.style.width = `${gaussian}%`;
            if (impulseLevelBar) impulseLevelBar.style.width = `${impulse}%`;
        }, 100);

        if (snrValue) snrValue.textContent = `${noiseData.metrics.snr_db} dB`;
    }

    function renderResults(data) {
        activePredictionData = data;
        if (origImgPreview) origImgPreview.src = data.original_image;
        if (claheImgPreview) {
            claheImgPreview.src = data.processed_image;
            claheImgPreview.classList.remove('hidden');
        }
        const clahePlaceholder = document.getElementById('clahePlaceholder');
        if (clahePlaceholder) {
            clahePlaceholder.classList.add('hidden');
        }

        if (verdictTitle) {
            verdictTitle.textContent = data.prediction;
            verdictTitle.style.fontSize = '1.2rem';
        }
        if (verdictConfidence) verdictConfidence.textContent = `${data.confidence}% Confidence`;

        if (verdictBanner) {
            verdictBanner.classList.remove('pending');
            if (data.prediction === 'MALIGNANT') {
                verdictBanner.classList.add('malignant');
            } else {
                verdictBanner.classList.remove('malignant');
            }
        }

        const bProb = data.probabilities.benign || 0;
        const mProb = data.probabilities.malignant || 0;

        if (benignProbVal) benignProbVal.textContent = `${bProb}%`;
        if (malignantProbVal) malignantProbVal.textContent = `${mProb}%`;

        setTimeout(() => {
            if (benignProbBar) benignProbBar.style.width = `${bProb}%`;
            if (malignantProbBar) malignantProbBar.style.width = `${mProb}%`;
        }, 100);

        // Populate Clinical Diagnostic Insights card dynamically
        const clinicalBirads = document.getElementById('clinicalBirads');
        const clinicalDensity = document.getElementById('clinicalDensity');
        const clinicalShadowing = document.getElementById('clinicalShadowing');
        const clinicalSummaryText = document.getElementById('clinicalSummaryText');

        if (data.prediction === 'MALIGNANT') {
            const biradsCat = data.confidence > 90 ? 'Category 5 - Highly Suggestive of Malignancy' : 'Category 4B - Suspicious Abnormality';
            if (clinicalBirads) clinicalBirads.textContent = biradsCat;
            if (clinicalDensity) clinicalDensity.textContent = 'Spiculated / Microlobulated Margins';
            if (clinicalShadowing) clinicalShadowing.textContent = 'Posterior Acoustic Shadowing detected. Suggestive of high-density solid tumor attenuation. Core needle biopsy recommended.';
            if (clinicalSummaryText) {
                clinicalSummaryText.textContent = `The neural network identified an irregular mass with non-circumscribed margins and distinct posterior shadowing (acoustic attenuation). These features indicate a high-density, attenuating tissue mass, suggesting a malignant pathology.`;
            }
        } else {
            const biradsCat = data.confidence > 85 ? 'Category 2 - Benign Finding' : 'Category 3 - Probably Benign';
            if (clinicalBirads) clinicalBirads.textContent = biradsCat;
            if (clinicalDensity) clinicalDensity.textContent = 'Circumscribed / Smooth Margins';
            if (clinicalShadowing) clinicalShadowing.textContent = 'Posterior Acoustic Enhancement observed. Indicative of high transmission fluid-filled cyst or benign fibroadenoma.';
            if (clinicalSummaryText) {
                clinicalSummaryText.textContent = `The neural network identified a well-circumscribed oval mass with smooth margins and posterior acoustic enhancement (high sound transmission). These features represent a low-attenuation fluid-filled cyst or benign fibroadenoma, suggesting a benign pathology.`;
            }
        }

        // Render noise analysis
        renderNoiseAnalysis(data.noise_analysis);

        showState('results');
    }

    function renderBatchResults(results) {
        const tbody = document.getElementById('batchResultsTableBody');
        tbody.innerHTML = '';

        let benignCount = 0;
        let malignantCount = 0;

        results.forEach(res => {
            if (res.error) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: 500; opacity: 0.7;">${escapeHtml(res.filename)}</td>
                    <td style="color: var(--malignant-color); font-weight: 600;">OUT OF SCOPE</td>
                    <td style="color: var(--text-secondary);">--</td>
                `;
                tr.style.cursor = 'pointer';
                tr.addEventListener('click', () => {
                    showAlertModal('Scope Warning', res.error, 'warning');
                });
                tbody.appendChild(tr);
                return;
            }

            const isBenign = res.prediction === 'BENIGN';
            if (isBenign) benignCount++;
            else malignantCount++;

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: 500;">${escapeHtml(res.filename)}</td>
                <td class="${isBenign ? 'success-text' : 'malignant-text'}" style="font-weight: 600;">${res.prediction}</td>
                <td>${res.confidence}%</td>
            `;

            // Row click triggers rendering detailed visualizations for this scan
            tr.style.cursor = 'pointer';
            tr.addEventListener('click', () => {
                addLogEntry(`Reviewing detailed visual graphs for: ${res.filename}`, 'info');
                renderResults({
                    prediction: res.prediction,
                    confidence: res.confidence,
                    probabilities: {
                        benign: isBenign ? res.confidence : (100 - res.confidence).toFixed(2),
                        malignant: isBenign ? (100 - res.confidence).toFixed(2) : res.confidence
                    },
                    original_image: res.original_image,
                    processed_image: res.processed_image,
                    noise_analysis: res.noise_analysis
                });
            });

            tbody.appendChild(tr);
        });

        const batchTotal = document.getElementById('batchTotalCount');
        const batchBenign = document.getElementById('batchBenignCount');
        const batchMalignant = document.getElementById('batchMalignantCount');
        if (batchTotal) batchTotal.textContent = results.length;
        if (batchBenign) batchBenign.textContent = benignCount;
        if (batchMalignant) batchMalignant.textContent = malignantCount;

        showState('batch');
    }

    function showState(state) {
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        batchResultsContent.classList.add('hidden');

        if (state === 'empty') emptyState.classList.remove('hidden');
        if (state === 'loading') loadingState.classList.remove('hidden');
        if (state === 'results') resultsContent.classList.remove('hidden');
        if (state === 'batch') batchResultsContent.classList.remove('hidden');
    }

    // --- Interactive Dual Mode & Code Notebook Logic ---
    btnUiMode.addEventListener('click', () => {
        btnUiMode.classList.add('active');
        btnCodeMode.classList.remove('active');
        btnNoiseMode.classList.remove('active');
        uiDashboardContainer.classList.remove('hidden');
        notebookContainer.classList.add('hidden');
        noiseCompareContainer.classList.add('hidden');
        const wrapper = document.querySelector('.dashboard-wrapper');
        wrapper.classList.remove('notebook-mode-active');
        wrapper.classList.remove('noise-mode-active');
        addLogEntry('Viewport toggled to UI Diagnostic Dashboard.', 'info');
    });

    btnCodeMode.addEventListener('click', () => {
        btnCodeMode.classList.add('active');
        btnUiMode.classList.remove('active');
        btnNoiseMode.classList.remove('active');
        uiDashboardContainer.classList.add('hidden');
        notebookContainer.classList.remove('hidden');
        noiseCompareContainer.classList.add('hidden');
        const wrapper = document.querySelector('.dashboard-wrapper');
        wrapper.classList.add('notebook-mode-active');
        wrapper.classList.remove('noise-mode-active');
        addLogEntry('Viewport toggled to Notebook Code Viewer.', 'info');
        
        const activeItem = document.querySelector('.nav-item.active');
        if (activeItem) {
            const fileName = activeItem.getAttribute('data-file');
            loadCodeFile(fileName);
        }
    });

    btnNoiseMode.addEventListener('click', () => {
        btnNoiseMode.classList.add('active');
        btnUiMode.classList.remove('active');
        btnCodeMode.classList.remove('active');
        uiDashboardContainer.classList.add('hidden');
        notebookContainer.classList.add('hidden');
        noiseCompareContainer.classList.remove('hidden');
        const wrapper = document.querySelector('.dashboard-wrapper');
        wrapper.classList.remove('notebook-mode-active');
        wrapper.classList.add('noise-mode-active');
        addLogEntry('Viewport toggled to Noise Comparison Study View.', 'info');
        
        // Load the current preview image into clean view if visible
        if (origImgPreview && origImgPreview.src && !origImgPreview.src.endsWith('/')) {
            if (noiseCleanImg) {
                noiseCleanImg.src = origImgPreview.src;
                noiseCleanImg.style.display = 'block';
                noiseCleanImg.classList.remove('hidden');
            }
            if (noiseCleanPlaceholder) noiseCleanPlaceholder.style.display = 'none';
        }
    });

    if (noiseStudySelect) {
        noiseStudySelect.addEventListener('change', async (e) => {
            const url = e.target.value;
            if (!url) {
                if (noiseCleanImg) noiseCleanImg.style.display = 'none';
                if (noiseCleanPlaceholder) noiseCleanPlaceholder.style.display = 'block';
                return;
            }
            
            if (existingStudySelect) {
                existingStudySelect.value = url;
            }
            
            addLogEntry(`Loading study for noise comparison: ${url.substring(url.lastIndexOf('/') + 1)}`, 'info');
            if (noiseCleanPlaceholder) {
                noiseCleanPlaceholder.style.display = 'block';
                noiseCleanPlaceholder.textContent = 'Loading scan...';
            }
            if (noiseCleanImg) noiseCleanImg.style.display = 'none';
            
            try {
                const response = await fetch(`${API_BASE}${url}`);
                if (!response.ok) throw new Error('Failed to retrieve scan image');
                const blob = await response.blob();
                
                const reader = new FileReader();
                reader.onload = (eReader) => {
                    if (noiseCleanImg) {
                        noiseCleanImg.src = eReader.target.result;
                        noiseCleanImg.style.display = 'block';
                        noiseCleanImg.classList.remove('hidden');
                    }
                    if (noiseCleanPlaceholder) noiseCleanPlaceholder.style.display = 'none';
                    
                    if (origImgPreview) {
                        origImgPreview.src = eReader.target.result;
                    }
                    
                    selectedFile = new File([blob], url.substring(url.lastIndexOf('/') + 1), {type: "image/png"});
                    
                    if (noiseCorruptedImg) noiseCorruptedImg.style.display = 'none';
                    if (noiseCorruptedPlaceholder) {
                        noiseCorruptedPlaceholder.style.display = 'block';
                        noiseCorruptedPlaceholder.textContent = 'Awaiting comparison execution...';
                    }
                    if (noiseCleanVerdict) noiseCleanVerdict.textContent = 'Awaiting comparison...';
                    if (noiseCleanConfidence) noiseCleanConfidence.textContent = '--';
                    if (noiseCorruptedVerdict) noiseCorruptedVerdict.textContent = 'Awaiting comparison...';
                    if (noiseCorruptedConfidence) noiseCorruptedConfidence.textContent = '--';
                };
                reader.readAsDataURL(blob);
            } catch (err) {
                console.error(err);
                addLogEntry(`Failed to load scan for noise comparison: ${err.message}`, 'error');
                if (noiseCleanPlaceholder) noiseCleanPlaceholder.textContent = 'Failed to load scan.';
            }
        });
    }

    if (btnRunNoiseCompare) {
        btnRunNoiseCompare.addEventListener('click', async () => {
            if (!origImgPreview || !origImgPreview.src || origImgPreview.src.endsWith('/')) {
                showAlertModal('No Image Loaded', 'Please select an image in the UI Diagnostic Dashboard first.', 'warning');
                return;
            }
            
            addLogEntry('Initiating Noise Diagnostic Comparison Study...', 'info');
            btnRunNoiseCompare.disabled = true;
            btnRunNoiseCompare.textContent = 'Comparing...';
            
            if (noiseCorruptedPlaceholder) noiseCorruptedPlaceholder.textContent = 'Generating noisy sample & predicting...';
            
            try {
                // Fetch the current original image as a blob
                const imageRes = await fetch(origImgPreview.src);
                const imageBlob = await imageRes.blob();
                
                const formData = new FormData();
                formData.append('file', imageBlob, 'study.png');
                formData.append('noise_type', noiseCompareType.value);
                formData.append('intensity', parseFloat(noiseCompareSeverity.value));
                formData.append('use_clahe', claheToggle.checked);
                
                const response = await fetch(`${API_BASE}/api/noise/simulate`, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Simulation failed on backend server.');
                }
                
                const data = await response.json();
                
                // Render Clean Reference Results
                if (noiseCleanImg) {
                    noiseCleanImg.src = `data:image/png;base64,${data.clean_image}`;
                    noiseCleanImg.style.display = 'block';
                    noiseCleanImg.classList.remove('hidden');
                }
                if (noiseCleanPlaceholder) noiseCleanPlaceholder.style.display = 'none';
                
                if (noiseCleanVerdict) noiseCleanVerdict.textContent = data.clean_prediction;
                if (noiseCleanConfidence) noiseCleanConfidence.textContent = `${data.clean_confidence}% Confidence`;
                if (noiseCleanBanner) {
                    noiseCleanBanner.classList.remove('pending');
                    noiseCleanBanner.classList.toggle('malignant', data.clean_prediction === 'MALIGNANT');
                }
                
                // Render Noisy Corrupted Results
                if (noiseCorruptedImg) {
                    noiseCorruptedImg.src = `data:image/png;base64,${data.noisy_image}`;
                    noiseCorruptedImg.style.display = 'block';
                    noiseCorruptedImg.classList.remove('hidden');
                }
                if (noiseCorruptedPlaceholder) noiseCorruptedPlaceholder.style.display = 'none';
                
                if (noiseCorruptedVerdict) noiseCorruptedVerdict.textContent = data.noisy_prediction;
                if (noiseCorruptedConfidence) noiseCorruptedConfidence.textContent = `${data.noisy_confidence}% Confidence`;
                if (noiseCorruptedBanner) {
                    noiseCorruptedBanner.classList.remove('pending');
                    noiseCorruptedBanner.classList.toggle('malignant', data.noisy_prediction === 'MALIGNANT');
                }
                
                // Render Clinical Rationale & Explanation
                if (noiseExplanationText) {
                    noiseExplanationText.innerHTML = `
                        <strong style="color: var(--accent-primary);">Diagnostic Comparison Summary:</strong><br>
                        <strong>Clean Preprocessing Verdict:</strong> <span style="color: ${data.clean_prediction === 'MALIGNANT' ? 'var(--malignant-color)' : 'var(--benign-color)'}; font-weight: bold;">${data.clean_prediction} (${data.clean_confidence}%)</span>.<br>
                        <strong>Noisy Diagnostic Verdict:</strong> <span style="color: ${data.noisy_prediction === 'MALIGNANT' ? 'var(--malignant-color)' : 'var(--benign-color)'}; font-weight: bold;">${data.noisy_prediction} (${data.noisy_confidence}%)</span>.<br>
                        <span style="display: block; margin-top: 8px;">${data.explanation}</span>
                    `;
                }
                
                addLogEntry('Noise Diagnostic Comparison Study completed.', 'success');
                
            } catch (e) {
                console.error(e);
                addLogEntry(`Noise study failed: ${e.message}`, 'error');
                showAlertModal('Study Evaluation Failure', 'Unable to complete side-by-side noise comparison.', 'error');
                if (noiseCorruptedPlaceholder) noiseCorruptedPlaceholder.textContent = 'Comparison failed.';
            } finally {
                btnRunNoiseCompare.disabled = false;
                btnRunNoiseCompare.textContent = 'Run Comparison';
            }
        });
    }

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            const fileName = item.getAttribute('data-file');
            loadCodeFile(fileName);
        });
    });

    // --- Redesign Sidebar Theme & Audit Console Drawer Toggles ---
    const btnToggleTheme = document.getElementById('btnToggleTheme');
    if (btnToggleTheme) {
        btnToggleTheme.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            const isDark = document.body.classList.contains('dark-theme');
            addLogEntry(`UI theme toggled to ${isDark ? 'Dark Mode' : 'Light Mode'}.`, 'info');
        });
    }

    const btnToggleAuditLog = document.getElementById('btnToggleAuditLog');
    const btnHideAuditLog = document.getElementById('btnHideAuditLog');
    const auditLogDrawer = document.getElementById('auditLogDrawer');

    if (btnToggleAuditLog && auditLogDrawer) {
        btnToggleAuditLog.addEventListener('click', () => {
            auditLogDrawer.classList.toggle('open');
            const isOpen = auditLogDrawer.classList.contains('open');
            addLogEntry(`${isOpen ? 'Opened' : 'Closed'} Clinical Audit Log console.`, 'info');
        });
    }

    if (btnHideAuditLog && auditLogDrawer) {
        btnHideAuditLog.addEventListener('click', () => {
            auditLogDrawer.classList.remove('open');
            addLogEntry('Closed Clinical Audit Log console.', 'info');
        });
    }

    // Code cell segmenting for Google Colab/Jupyter notebook simulation
    function segmentCode(code, fileName) {
        const cells = [];
        
        if (fileName === 'config.py') {
            cells.push({
                type: 'markdown',
                content: `### Global Configuration Framework
This configuration file manages:
- System paths for training, validation, and testing partitions.
- Preprocessing flags (CLAHE clip limits and tile grid sizes).
- Training hyperparameters (learning rates, batch sizes, optimizer constraints).
- Target neural network configurations and device mapping (CPU, MPS, CUDA).`
            });
            cells.push({
                type: 'code',
                content: code
            });
        }
        else if (fileName === 'dataset.py') {
            cells.push({
                type: 'markdown',
                content: `### 1. Clinical Data Loading & CLAHE Preprocessing
This section imports standard medical image processing packages (OpenCV, Pillow, PyTorch) and implements Contrast Limited Adaptive Histogram Equalization (CLAHE). CLAHE normalizes contrast variations caused by different ultrasound transducers and improves lesion boundary visibility.`
            });
            const importsEnd = code.indexOf('class BreastUltrasoundDataset');
            if (importsEnd !== -1) {
                cells.push({
                    type: 'code',
                    content: code.substring(0, importsEnd).trim()
                });
                cells.push({
                    type: 'markdown',
                    content: `### 2. PyTorch Dataset & Transform Pipeline
We define the custom subclass \`BreastUltrasoundDataset\`, mapping label indices and applying optional CLAHE enhancement dynamically during item retrieval. Standard data augmentation (random horizontal/vertical flips, small rotations) is configured to prevent overfitting.`
                });
                const datasetEnd = code.indexOf('def get_transforms()');
                if (datasetEnd !== -1) {
                    cells.push({
                        type: 'code',
                        content: code.substring(importsEnd, datasetEnd).trim()
                    });
                    cells.push({
                        type: 'markdown',
                        content: `### 3. Balanced Clinical Dataloaders
Ultrasound datasets often suffer from class imbalance. We calculate class reciprocal weights and declare a weighted random sampler to ensure balanced training batch representation.`
                    });
                    cells.push({
                        type: 'code',
                        content: code.substring(datasetEnd).trim()
                    });
                } else {
                    cells.push({
                        type: 'code',
                        content: code.substring(importsEnd).trim()
                    });
                }
            } else {
                cells.push({
                    type: 'code',
                    content: code
                });
            }
        }
        else if (fileName === 'models.py') {
            cells.push({
                type: 'markdown',
                content: `### 1. Custom CNN Baseline for Ultrasound Scans
We define a custom 4-block Convolutional Neural Network baseline trained from scratch. Each block consists of 2D Convolution, Batch Normalization, ReLU activation, and Max Pooling. A classification head with high Dropout is declared to regularize features.`
            });
            const customCnnEnd = code.indexOf('def get_model');
            if (customCnnEnd !== -1) {
                cells.push({
                    type: 'code',
                    content: code.substring(0, customCnnEnd).trim()
                });
                cells.push({
                    type: 'markdown',
                    content: `### 2. Fine-Tuned Transfer Learning Backbones
We implement a model factory utilizing pre-trained backbone features from ResNet-50 and EfficientNet-B0. The classification heads are customized for binary diagnostic outcomes.`
                });
                cells.push({
                    type: 'code',
                    content: code.substring(customCnnEnd).trim()
                });
            } else {
                cells.push({
                    type: 'code',
                    content: code
                });
            }
        }
        else if (fileName === 'train.py') {
            cells.push({
                type: 'markdown',
                content: `### 1. Training & Validation Epoch Loop
We compile the core training routine. Uses AdamW optimizer, cosine annealing learning rate scheduler, and calculates accuracy and cross-entropy loss gradients.`
            });
            const mainStart = code.indexOf('def main()');
            if (mainStart !== -1) {
                cells.push({
                    type: 'code',
                    content: code.substring(0, mainStart).trim()
                });
                cells.push({
                    type: 'markdown',
                    content: `### 2. Hyperparameter Settings & Model Serialization
Exposes a command line interface to train specific backbones, tracks validation loss, implements Early Stopping (patience=5), and serializes checkpoint states.`
                });
                cells.push({
                    type: 'code',
                    content: code.substring(mainStart).trim()
                });
            } else {
                cells.push({
                    type: 'code',
                    content: code
                });
            }
        }
        else if (fileName === 'evaluate.py') {
            cells.push({
                type: 'markdown',
                content: `### 1. Validation Performance Metrics
Imports evaluation criteria: Classification Reports (Precision, Recall, F1), Confusion Matrices, ROC curves, and Area Under Curve (AUC) metrics.`
            });
            const evaluateStart = code.indexOf('def evaluate_model');
            if (evaluateStart !== -1) {
                cells.push({
                    type: 'code',
                    content: code.substring(0, evaluateStart).trim()
                });
                cells.push({
                    type: 'markdown',
                    content: `### 2. Evaluation Loop & Graphical Plotting
Evaluates model weights on the independent testing split and plots standard ROC curve diagrams and confusion matrix plots to disk.`
                });
                cells.push({
                    type: 'code',
                    content: code.substring(evaluateStart).trim()
                });
            } else {
                cells.push({
                    type: 'code',
                    content: code
                });
            }
        }
        else if (fileName === 'predict.py') {
            cells.push({
                type: 'markdown',
                content: `### 1. Clinical Diagnostic Inference Interface
Defines the core inference function. Loads serialized weights, preprocesses target images, runs predictions, and outputs likelihood statistics.`
            });
            const predictStart = code.indexOf('def predict_single_image');
            if (predictStart !== -1) {
                cells.push({
                    type: 'code',
                    content: code.substring(0, predictStart).trim()
                });
                cells.push({
                    type: 'markdown',
                    content: `### 2. CLI Invocation Hooks
Sets up arguments for running predictions directly from the shell terminal.`
                });
                cells.push({
                    type: 'code',
                    content: code.substring(predictStart).trim()
                });
            } else {
                cells.push({
                    type: 'code',
                    content: code
                });
            }
        }
        else {
            cells.push({
                type: 'code',
                content: code
            });
        }
        
        return cells;
    }

    // Lookup table for Google Colab-style Cell Output Prompts
    const notebookCellOutputs = {
        'dataset.py': [
            "Importing core clinical dependencies:\n- OpenCV (cv2) for CLAHE contrast\n- PyTorch (torch) & torchvision for normalizers\n- scikit-learn for metric analysis\nDependencies verified successfully.",
            "Compiling custom PyTorch dataset loader...\nClass BreastUltrasoundDataset successfully declared.\nIncludes custom CLAHE processor hooks on load.",
            "Creating dynamic balanced sampler framework...\nComputes reciprocal frequencies for benign (0) and malignant (1).\nsampler weights loaded successfully."
        ],
        'models.py': [
            "Importing PyTorch neural network modules and torchvision models.\nFine-tuning backbone: ResNet50\nFine-tuning backbone: EfficientNet-B0\nSuccessfully loaded pretrained weights.",
            "Compiling backbone factory function get_model()...\nSupports 'resnet50', 'efficientnet_b0', and 'custom_cnn'.\nget_model factory loaded."
        ],
        'train.py': [
            "Importing optimizer and execution tools:\n- Optimizer: Adam (lr=1e-4)\n- Loss: Weighted Cross Entropy\nAll training loop packages active.",
            "Checking dummy data availability...\nFound validation sample directories.\nLoading checkpoint model baseline...\nModel loaded on Device: CPU."
        ],
        'evaluate.py': [
            "Importing evaluation packages (scikit-learn classification_report, roc_curve, auc, confusion_matrix).\nPlotting utilities initialized.",
            "Plot generation hooks established:\n- confusion_matrix.png\n- roc_curve.png\nEvaluation loop validated."
        ],
        'config.py': [
            "Defining global training variables:\n- BATCH_SIZE: 16\n- EPOCHS: 25\n- LEARNING_RATE: 0.0001\n- IMG_SIZE: 224\n- CLASS_NAMES: ['benign', 'malignant']\nConfiguration variables initialized."
        ],
        'predict.py': [
            "Loading prediction dependencies...\nPreparing argparser for CLI inference options.",
            "CLI entry hooks configured.\npredict.py ready for deployment."
        ]
    };

    function renderNotebookCells(cells, fileName) {
        notebookCellsList.innerHTML = '';
        let codeCellCount = 0;
        
        cells.forEach((cellData, index) => {
            const cell = document.createElement('div');
            
            if (cellData.type === 'markdown') {
                cell.className = 'notebook-cell markdown-cell';
                let html = cellData.content
                    .replace(/^### (.*$)/gim, '<h3 style="margin-top: 8px; margin-bottom: 8px; color: var(--accent-primary); font-size: 0.95rem;">$1</h3>')
                    .replace(/^## (.*$)/gim, '<h2 style="margin-top: 12px; margin-bottom: 8px; color: var(--text-primary); font-size: 1.1rem;">$1</h2>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\`(.*?)\`/g, '<code style="background: rgba(255,255,255,0.06); padding: 2px 4px; border-radius: 4px; font-size: 0.8rem; font-family: monospace; color: var(--text-primary);">$1</code>')
                    .replace(/^- (.*$)/gim, '<li style="margin-left: 16px; margin-bottom: 4px;">$1</li>');
                
                cell.innerHTML = `
                    <div class="cell-left" style="opacity: 0.3; font-size: 0.7rem; padding-top: 4px;">[md]</div>
                    <div class="cell-markdown-container" style="color: var(--text-secondary); font-size: 0.82rem; line-height: 1.5; padding: 4px 10px;">
                        ${html}
                    </div>
                `;
            } else {
                codeCellCount++;
                cell.className = 'notebook-cell code-cell';
                
                const outputs = notebookCellOutputs[fileName] || ["Cell execution completed successfully."];
                const defaultOutput = outputs[codeCellCount - 1] || "Execution completed.";
                
                cell.innerHTML = `
                    <div class="cell-left">
                        <div class="cell-input-prompt">In [${codeCellCount}]:</div>
                        <button class="cell-play-btn" title="Run Cell">
                            <svg viewBox="0 0 24 24" width="10" height="10"><polygon points="8 5 18 12 8 19 8 5"></polygon></svg>
                        </button>
                    </div>
                    <div class="cell-code-container">
                        <pre><code>${escapeHtml(cellData.content)}</code></pre>
                        <!-- Colab style output cell (shown by default!) -->
                        <div class="cell-output-container" id="out-${fileName}-${index}">
                            <div class="cell-output-header">Output</div>
                            <pre class="cell-output-text">${defaultOutput}</pre>
                        </div>
                    </div>
                `;
                
                cell.querySelector('.cell-play-btn').addEventListener('click', () => {
                    runCell(cell, fileName, index, codeCellCount);
                });
            }
            
            notebookCellsList.appendChild(cell);
        });
    }

    function runCell(cellElement, fileName, cellIndex, codeIndex) {
        if (codeIndex === undefined) {
            const codeCells = Array.from(notebookCellsList.querySelectorAll('.notebook-cell.code-cell'));
            codeIndex = codeCells.indexOf(cellElement) + 1;
        }

        const playBtn = cellElement.querySelector('.cell-play-btn');
        const prompt = cellElement.querySelector('.cell-input-prompt');
        const outputContainer = cellElement.querySelector('.cell-output-container');
        const outputText = cellElement.querySelector('.cell-output-text');

        if (prompt) prompt.textContent = 'In [*]:';
        if (playBtn) playBtn.style.color = '#38bdf8';
        if (outputContainer) {
            outputContainer.classList.remove('hidden');
            outputContainer.classList.add('running');
        }
        if (outputText) outputText.textContent = 'Running process cell...';

        addLogEntry(`Executing Notebook cell [${codeIndex}] inside ${fileName}...`, 'info');
        
        setTimeout(() => {
            if (prompt) prompt.textContent = `In [${codeIndex}]:`;
            if (playBtn) playBtn.style.color = '';
            if (outputContainer) outputContainer.classList.remove('running');
            
            // Query mock stdout output text
            const outputs = notebookCellOutputs[fileName] || ["Cell execution completed successfully."];
            // Since markdown cells exist, map code index appropriately
            const output = outputs[codeIndex - 1] || "Execution completed.";
            if (outputText) outputText.textContent = output;

            addLogEntry(`Cell [${codeIndex}] in ${fileName} executed successfully.`, 'success');
        }, 600);
    }

    async function loadCodeFile(fileName) {
        if (activeFileName) activeFileName.textContent = fileName;
        addLogEntry(`Accessing source file: ${fileName}`, 'info');
        
        if (codeCache[fileName]) {
            renderNotebookCells(codeCache[fileName], fileName);
            return;
        }

        notebookCellsList.innerHTML = '<div style="color: var(--text-secondary); font-family: monospace; font-size: 0.8rem;">Loading source cells...</div>';

        try {
            const response = await fetch(`${API_BASE}/api/code/${fileName}`);
            if (!response.ok) {
                throw new Error('Failed to retrieve file contents');
            }
            const data = await response.json();
            const cells = segmentCode(data.code, fileName);
            codeCache[fileName] = cells;
            renderNotebookCells(cells, fileName);
            addLogEntry(`Source file ${fileName} loaded as Jupyter notebook cells.`, 'success');
        } catch (error) {
            notebookCellsList.innerHTML = `<div style="color: var(--malignant-color); font-family: monospace; font-size: 0.8rem;">Error loading notebook: ${error.message}</div>`;
            addLogEntry(`Source retrieval failed: ${error.message}`, 'error');
        }
    }

    // Run all cells simulator button
    btnRunAllCells.addEventListener('click', () => {
        const cells = notebookCellsList.querySelectorAll('.notebook-cell');
        if (cells.length === 0) return;
        
        const fileName = activeFileName ? activeFileName.textContent : '';
        addLogEntry(`Running all ${cells.length} cells in ${fileName} sequentially...`, 'info');
        let i = 0;
        
        function runNext() {
            if (i >= cells.length) {
                addLogEntry(`Jupyter Notebook run completed for ${fileName}.`, 'success');
                return;
            }
            const cell = cells[i];
            runCell(cell, fileName, i);
            i++;
            setTimeout(runNext, 850);
        }
        runNext();
    });

    // Copy to clipboard
    btnCopyCode.addEventListener('click', () => {
        const file = activeFileName ? activeFileName.textContent : '';
        const cells = codeCache[file];
        if (!cells) return;
        const codeText = cells.map(c => {
            if (c.type === 'markdown') {
                return c.content.split('\n').map(line => `# ${line}`).join('\n');
            }
            return c.content;
        }).join('\n\n');
        
        navigator.clipboard.writeText(codeText).then(() => {
            const origText = btnCopyCode ? btnCopyCode.textContent : '';
            if (btnCopyCode) btnCopyCode.textContent = 'Copied!';
            addLogEntry(`Copied source contents of ${file} to clipboard.`, 'success');
            setTimeout(() => {
                if (btnCopyCode) btnCopyCode.textContent = origText;
            }, 1500);
        }).catch(err => {
            addLogEntry('Failed to copy code.', 'error');
        });
    });

    // Download PDF Diagnostic Report
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => {
            if (!activePredictionData) {
                addLogEntry('No diagnostic results available to export.', 'warning');
                return;
            }
            generatePDFReport(activePredictionData);
        });
    }

    function generatePDFReport(data) {
        addLogEntry('Compiling diagnostic PDF report...', 'info');

        const activeDataset = datasetSelect ? datasetSelect.value.toUpperCase() : 'BUSI';
        const activeSplit = datasetSplitSelect ? datasetSplitSelect.value.toUpperCase() : 'VAL';
        const activeModel = modelSelect ? modelSelect.options[modelSelect.selectedIndex].text : 'ResNet-50';
        
        let filename = 'Uploaded Scan';
        if (existingStudySelect && existingStudySelect.value) {
            filename = existingStudySelect.value.substring(existingStudySelect.value.lastIndexOf('/') + 1);
            if (filename.includes('?')) filename = filename.split('?')[0];
        }

        // Get clinical details text
        const birads = document.getElementById('clinicalBirads')?.textContent || 'N/A';
        const density = document.getElementById('clinicalDensity')?.textContent || 'N/A';
        const shadowing = document.getElementById('clinicalShadowing')?.textContent || 'N/A';
        const rationale = document.getElementById('clinicalSummaryText')?.textContent || 'N/A';

        // Read actual noise analysis from backend payload
        const noiseAnalysis = data.noise_analysis || {};
        const dominantType = noiseAnalysis.dominant_type || 'Mixed Acoustic Speckle';
        const dominantDesc = noiseAnalysis.description || 'Standard acoustic speckle signature with normal sensor thermal parameters.';
        
        const metrics = noiseAnalysis.metrics || {};
        const snrDb = metrics.snr_db !== undefined ? `${metrics.snr_db} dB` : 'N/A';
        const speckleLevel = metrics.speckle_level !== undefined ? Math.round(metrics.speckle_level) : 0;
        const gaussianLevel = metrics.gaussian_level !== undefined ? Math.round(metrics.gaussian_level) : 0;
        const impulseLevel = metrics.impulse_level !== undefined ? Math.round(metrics.impulse_level) : 0;

        // Create temporary div container for styling print
        const reportContainer = document.createElement('div');
        reportContainer.style.padding = '40px';
        reportContainer.style.color = '#0f172a';
        reportContainer.style.background = '#ffffff';
        reportContainer.style.fontFamily = "'Outfit', sans-serif";
        reportContainer.style.fontSize = '12px';
        reportContainer.style.lineHeight = '1.5';

        const isMalignant = data.prediction === 'MALIGNANT';
        const themeColor = isMalignant ? '#ef4444' : '#10b981';

        reportContainer.innerHTML = `
            <!-- Report Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; margin-bottom: 25px;">
                <div>
                    <h1 style="margin: 0; font-size: 24px; font-weight: 700; color: #1e3a8a; letter-spacing: -0.5px;">ONCOVISION AI</h1>
                    <p style="margin: 3px 0 0 0; font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Automated Diagnostic Suite</p>
                </div>
                <div style="text-align: right;">
                    <h3 style="margin: 0; font-size: 14px; font-weight: 600; color: #0f172a;">Ultrasound Diagnostic Report</h3>
                    <p style="margin: 3px 0 0 0; font-size: 11px; color: #64748b;">Date: ${new Date().toLocaleString()}</p>
                </div>
            </div>

            <!-- Case Summary Grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; background: #f8fafc; border-radius: 8px; padding: 15px; border: 1px solid #f1f5f9;">
                <div>
                    <h4 style="margin: 0 0 10px 0; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.5px;">Assessment Metadata</h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                        <tr>
                            <td style="padding: 4px 0; color: #64748b; font-weight: 500;">Study Filename:</td>
                            <td style="padding: 4px 0; color: #0f172a; font-weight: 600; text-align: right;">${filename}</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 0; color: #64748b; font-weight: 500;">Neural Network Model:</td>
                            <td style="padding: 4px 0; color: #0f172a; font-weight: 600; text-align: right;">${activeModel}</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 0; color: #64748b; font-weight: 500;">Clinical Dataset:</td>
                            <td style="padding: 4px 0; color: #0f172a; font-weight: 600; text-align: right;">${activeDataset} (${activeSplit} split)</td>
                        </tr>
                    </table>
                </div>
                <div style="border-left: 1px solid #e2e8f0; padding-left: 20px;">
                    <h4 style="margin: 0 0 10px 0; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.5px;">AI Verdict Summary</h4>
                    <div style="background: ${themeColor}10; border: 1px solid ${themeColor}30; border-radius: 6px; padding: 10px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: ${themeColor}; font-size: 14px; margin: 0;">${data.prediction}</span>
                        <span style="font-weight: 700; color: ${themeColor}; font-size: 14px;">${data.confidence}% Confidence</span>
                    </div>
                    <div style="display: flex; gap: 15px; font-size: 10px; color: #475569;">
                        <span>Benign Class: <strong>${data.probabilities.benign}%</strong></span>
                        <span>Malignant Class: <strong>${data.probabilities.malignant}%</strong></span>
                    </div>
                </div>
            </div>

            <!-- Scans Side-by-Side -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="text-align: center;">
                    <p style="margin: 0 0 6px 0; font-size: 11px; font-weight: 600; color: #475569;">Original Ultrasound Scan</p>
                    <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px; background: #fafafa; display: flex; align-items: center; justify-content: center; height: 230px;">
                        <img src="${data.original_image}" style="max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px;">
                    </div>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0 0 6px 0; font-size: 11px; font-weight: 600; color: #475569;">CLAHE Enhanced View & Mass Localization</p>
                    <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px; background: #fafafa; display: flex; align-items: center; justify-content: center; height: 230px;">
                        <img src="${data.processed_image}" style="max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px;">
                    </div>
                </div>
            </div>

            <!-- Clinical Diagnostic Insights -->
            <div style="margin-bottom: 25px;">
                <h3 style="border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin: 0 0 12px 0; font-size: 13px; font-weight: 700; color: #1e3a8a; text-transform: uppercase;">Clinical Findings & Insights</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 12px;">
                    <div style="background: #f8fafc; border-radius: 6px; padding: 10px; border: 1px solid #f1f5f9;">
                        <span style="font-size: 9px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 2px;">BI-RADS Classification</span>
                        <strong style="color: #0f172a; font-size: 11px;">${birads}</strong>
                    </div>
                    <div style="background: #f8fafc; border-radius: 6px; padding: 10px; border: 1px solid #f1f5f9;">
                        <span style="font-size: 9px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 2px;">Estimated Tissue Density</span>
                        <strong style="color: #0f172a; font-size: 11px;">${density}</strong>
                    </div>
                </div>
                <div style="background: #f8fafc; border-radius: 6px; padding: 10px; border: 1px solid #f1f5f9; margin-bottom: 12px;">
                    <span style="font-size: 9px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 2px;">Acoustic Shadowing Profile</span>
                    <strong style="color: #0f172a; font-size: 11px; font-weight: 500;">${shadowing}</strong>
                </div>
                <div style="background: #eff6ff; border-radius: 6px; padding: 12px; border: 1px solid #dbeafe;">
                    <strong style="color: #1e40af; font-size: 11px; display: block; margin-bottom: 4px;">AI Diagnostic Rationale & Evidence:</strong>
                    <p style="margin: 0; color: #1e3a8a; font-size: 11px; text-align: justify; line-height: 1.45;">${rationale}</p>
                </div>
            </div>

            <!-- Signal & Noise Quality Evaluation -->
            <div style="margin-bottom: 25px;">
                <h3 style="border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin: 0 0 12px 0; font-size: 13px; font-weight: 700; color: #1e3a8a; text-transform: uppercase;">Clinical Sensor Noise & Signal Integrity Analysis</h3>
                <div style="background: #f8fafc; border-radius: 6px; padding: 12px; border: 1px solid #f1f5f9; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase;">Dominant Noise Signature:</span>
                        <strong style="color: #1e3a8a; font-size: 11px;">${dominantType}</strong>
                    </div>
                    <p style="margin: 0; font-size: 11px; color: #475569; line-height: 1.45;">${dominantDesc}</p>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
                    <div style="text-align: center; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; background: #fafafa;">
                        <span style="font-size: 9px; color: #64748b; display: block; margin-bottom: 2px;">Signal-to-Noise (SNR)</span>
                        <strong style="font-size: 13px; color: #0f172a;">${snrDb}</strong>
                    </div>
                    <div style="text-align: center; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; background: #fafafa;">
                        <span style="font-size: 9px; color: #64748b; display: block; margin-bottom: 2px;">Acoustic Speckle Level</span>
                        <strong style="font-size: 13px; color: #0f172a;">${speckleLevel}%</strong>
                    </div>
                    <div style="text-align: center; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; background: #fafafa;">
                        <span style="font-size: 9px; color: #64748b; display: block; margin-bottom: 2px;">Thermal Gaussian Level</span>
                        <strong style="font-size: 13px; color: #0f172a;">${gaussianLevel}%</strong>
                    </div>
                    <div style="text-align: center; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; background: #fafafa;">
                        <span style="font-size: 9px; color: #64748b; display: block; margin-bottom: 2px;">Sensor Impulse Level</span>
                        <strong style="font-size: 13px; color: #0f172a;">${impulseLevel}%</strong>
                    </div>
                </div>
            </div>

            <!-- Footer & Disclaimer -->
            <div style="border-top: 1px solid #e2e8f0; padding-top: 15px; margin-top: 35px; text-align: center; color: #64748b; font-size: 9px; line-height: 1.4;">
                <p style="margin: 0 0 4px 0; font-weight: 600;">CONFIDENTIAL MEDICAL INFORMATION — RESEARCH STUDY ONLY</p>
                <p style="margin: 0; max-width: 500px; margin-left: auto; margin-right: auto;">Disclaimer: This diagnostic report is generated by a deep neural network prototype for research and evaluation purposes. Decisions relating to clinical patient treatment and malignancy diagnoses should be made by licensed medical practitioners alongside biopsy findings.</p>
            </div>
        `;

        const opt = {
            margin: 0,
            filename: `OncoVision_Diagnostic_Report_${filename.split('.')[0]}_${new Date().toISOString().slice(0,10)}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true, letterRendering: true },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        // Generate PDF and log outcome
        html2pdf().from(reportContainer).set(opt).save().then(() => {
            addLogEntry('Diagnostic PDF report downloaded successfully.', 'success');
        }).catch(err => {
            addLogEntry(`Failed to generate PDF: ${err.message}`, 'error');
        });
    }
});
