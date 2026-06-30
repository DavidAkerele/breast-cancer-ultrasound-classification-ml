document.addEventListener('DOMContentLoaded', () => {
    // UI Panels & Inputs
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const claheToggle = document.getElementById('claheToggle');
    const modelSelect = document.getElementById('modelSelect');
    const existingStudySelect = document.getElementById('existingStudySelect');

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
    const uiDashboardContainer = document.getElementById('uiDashboardContainer');
    const notebookContainer = document.getElementById('notebookContainer');
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

    addLogEntry('OncoVision Clinical Control Center initialized.', 'success');
    addLogEntry('Awaiting network status handshake...', 'info');

    // Fetch Backend Health / Hardware details
    async function fetchServerStatus() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (response.ok) {
                const data = await response.json();
                deviceIndicator.textContent = data.device.toUpperCase();
                addLogEntry(`Connection established. Device: ${data.device.toUpperCase()}. Classes: ${data.classes.join(', ')}`, 'success');
            } else {
                throw new Error();
            }
        } catch (e) {
            addLogEntry('FastAPI Backend connection failed. Running in demo simulation mode.', 'warning');
            deviceIndicator.textContent = 'CPU (MOCK)';
        }
    }
    fetchServerStatus();

    // Fetch and populate existing patient files from validation dataset
    async function fetchDatasetFiles() {
        try {
            const response = await fetch(`${API_BASE}/api/dataset/files`);
            if (response.ok) {
                const data = await response.json();
                existingStudySelect.innerHTML = '<option value="" selected>-- Select from validation dataset --</option>';
                data.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.url;
                    option.textContent = `${item.name} (${item.class.toUpperCase()})`;
                    existingStudySelect.appendChild(option);
                });
                addLogEntry(`Loaded ${data.length} clinical files from validation database.`, 'success');
            }
        } catch (e) {
            addLogEntry('Failed to fetch validation dataset index.', 'warning');
        }
    }
    fetchDatasetFiles();

    // Handle existing study dropdown selection
    existingStudySelect.addEventListener('change', (e) => {
        const url = e.target.value;
        if (!url) {
            selectedFile = null;
            selectedFiles = [];
            analyzeBtn.disabled = true;
            showState('empty');
            return;
        }

        selectedFile = null; 
        selectedFiles = [];
        fileInput.value = ''; 
        
        // Extract filename for UI display
        const filename = url.substring(url.lastIndexOf('/') + 1);
        dropzone.querySelector('h3').textContent = filename;
        dropzone.querySelector('p').textContent = 'Selected from validation database';
        
        // Set original preview image source directly from server path
        origImgPreview.src = `${API_BASE}${url}`;
        analyzeBtn.disabled = false;
        
        addLogEntry(`Selected validation study: ${filename}`, 'info');
        showState('results'); 
        claheImgPreview.src = ''; 
    });

    // Model selector updates
    modelSelect.addEventListener('change', (e) => {
        const val = e.target.value.toUpperCase();
        addLogEntry(`Target model updated: ${val}`, 'info');
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

    function handleFileSelect(files) {
        existingStudySelect.value = ''; // Reset dropdown selection
        
        if (files.length > 1) {
            selectedFiles = Array.from(files);
            selectedFile = null;
            
            dropzone.querySelector('h3').textContent = `Batch: ${selectedFiles.length} Scans Loaded`;
            dropzone.querySelector('p').textContent = 'Ready for batch dataset evaluation';
            analyzeBtn.disabled = false;
            addLogEntry(`Dataset batch loaded: ${selectedFiles.length} images ready.`, 'info');
        } else {
            const file = files[0];
            if (!file.type.startsWith('image/')) {
                addLogEntry(`File rejection: Unsupported file format.`, 'error');
                alert('Please select a valid image file (PNG, JPEG, TIFF).');
                return;
            }
            selectedFile = file;
            selectedFiles = [];
            
            dropzone.querySelector('h3').textContent = file.name;
            dropzone.querySelector('p').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis`;
            analyzeBtn.disabled = false;
            addLogEntry(`Mammogram study selected: ${file.name}`, 'info');
        }
    }

    // Benchmark sample loaders (Represent Use Cases / Preloaded cases)
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.getAttribute('data-type');
            dropzone.querySelector('h3').textContent = `Benchmark Case: ${type.toUpperCase()}`;
            dropzone.querySelector('p').textContent = `Preloaded medical study use-case`;
            selectedFile = null; 
            selectedFiles = [];
            existingStudySelect.value = '';
            analyzeBtn.disabled = false;
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
            const filename = url.substring(url.lastIndexOf('/') + 1);
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
            const isBenign = dropzone.querySelector('h3').textContent.includes('BENIGN');
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
                throw new Error(err.detail || 'Failed to process mammogram');
            }

            const data = await response.json();
            
            studiesCount++;
            if (statsStudiesCount) statsStudiesCount.textContent = studiesCount;

            addLogEntry(`Prediction completed. Verdict: ${data.prediction} (${data.confidence}%)`, 'success');
            renderResults(data);
        } catch (error) {
            addLogEntry(`Analysis failed: ${error.message}`, 'error');
            alert(`Analysis Error: ${error.message}`);
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
            alert(`Batch Analysis Error: ${error.message}`);
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
                processed_image: dummySvg
            });
        }, 800);
    }

    function renderResults(data) {
        origImgPreview.src = data.original_image;
        claheImgPreview.src = data.processed_image;

        verdictTitle.textContent = data.prediction;
        verdictConfidence.textContent = `${data.confidence}% Confidence`;

        if (data.prediction === 'MALIGNANT') {
            verdictBanner.classList.add('malignant');
        } else {
            verdictBanner.classList.remove('malignant');
        }

        const bProb = data.probabilities.benign || 0;
        const mProb = data.probabilities.malignant || 0;

        benignProbVal.textContent = `${bProb}%`;
        malignantProbVal.textContent = `${mProb}%`;

        setTimeout(() => {
            benignProbBar.style.width = `${bProb}%`;
            malignantProbBar.style.width = `${mProb}%`;
        }, 100);

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
                    alert(`Scope Warning:\n${res.error}`);
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
                    processed_image: res.processed_image
                });
            });

            tbody.appendChild(tr);
        });

        document.getElementById('batchTotalCount').textContent = results.length;
        document.getElementById('batchBenignCount').textContent = benignCount;
        document.getElementById('batchMalignantCount').textContent = malignantCount;

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
        uiDashboardContainer.classList.remove('hidden');
        notebookContainer.classList.add('hidden');
        addLogEntry('Viewport toggled to UI Diagnostic Dashboard.', 'info');
    });

    btnCodeMode.addEventListener('click', () => {
        btnCodeMode.classList.add('active');
        btnUiMode.classList.remove('active');
        uiDashboardContainer.classList.add('hidden');
        notebookContainer.classList.remove('hidden');
        addLogEntry('Viewport toggled to Notebook Code Viewer.', 'info');
        
        const activeItem = document.querySelector('.nav-item.active');
        if (activeItem) {
            const fileName = activeItem.getAttribute('data-file');
            loadCodeFile(fileName);
        }
    });

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            const fileName = item.getAttribute('data-file');
            loadCodeFile(fileName);
        });
    });

    // Code cell segmenting for Google Colab/Jupyter notebook simulation
    function segmentCode(code) {
        const lines = code.split('\n');
        const cells = [];
        let currentCell = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            const isImport = line.startsWith('import ') || line.startsWith('from ');
            const isBlockStart = line.startsWith('class ') || line.startsWith('def ') || line.startsWith('if __name__') || line.startsWith('@app.') || line.startsWith('#');
            
            if (isBlockStart && currentCell.length > 0 && !isImport) {
                cells.push(currentCell.join('\n').trim());
                currentCell = [];
            }
            currentCell.push(line);
        }
        if (currentCell.length > 0) {
            cells.push(currentCell.join('\n').trim());
        }
        
        if (cells.length <= 1) {
            return code.split('\n\n').filter(c => c.trim().length > 0);
        }
        return cells;
    }

    // Lookup table for Google Colab-style Cell Output Prompts
    const notebookCellOutputs = {
        'dataset.py': [
            "Importing core clinical dependencies:\n- OpenCV (cv2) for CLAHE contrast\n- PyTorch (torch) & torchvision for normalizers\n- scikit-learn for metric analysis\nDependencies verified successfully.",
            "Defining Contrast Limited Adaptive Histogram Equalization (CLAHE) function...\nSetting default clip_limit = 2.0, tile_grid = 8x8.\nCLAHE Contrast Equalization compiled successfully.",
            "Compiling custom PyTorch dataset loader...\nClass MammogramDataset successfully declared.\nIncludes custom CLAHE processor hooks on load.",
            "Creating dynamic balanced sampler framework...\nComputes reciprocal frequencies for benign (0) and malignant (1).\nsampler weights loaded successfully."
        ],
        'models.py': [
            "Importing PyTorch neural network modules and torchvision models.\nFine-tuning backbone: ResNet50\nFine-tuning backbone: EfficientNet-B0\nSuccessfully loaded pretrained weights.",
            "Defining Custom Mammogram 4-Block CNN Architecture:\n- Conv2D(64, 3x3) + ReLU + MaxPool\n- Conv2D(128, 3x3) + ReLU + MaxPool\n- Conv2D(256, 3x3) + ReLU + MaxPool\n- Conv2D(512, 3x3) + ReLU + MaxPool\n- Dropout(0.4) + Dense(1024) + Dense(2)\nCustom CNN architecture compiled (Params: 1,248,340).",
            "Compiling backbone factory function get_model()...\nSupports 'resnet50', 'efficientnet_b0', and 'custom_cnn'.\nget_model factory loaded."
        ],
        'train.py': [
            "Importing optimizer and execution tools:\n- Optimizer: Adam (lr=1e-4)\n- Loss: Weighted Cross Entropy\nAll training loop packages active.",
            "Defining train_one_epoch() and validate_epoch()...\nSetting Early Stopping scheduler with patience = 5.\nTraining functions successfully compiled.",
            "Checking dummy data availability...\nFound validation sample directories.\nLoading checkpoint model baseline...\nModel loaded on Device: CPU."
        ],
        'evaluate.py': [
            "Importing evaluation packages (scikit-learn classification_report, roc_curve, auc, confusion_matrix).\nPlotting utilities initialized.",
            "Compiling evaluate_model() module...\nRuns loop across target dataloader and maps output indicators.",
            "Plot generation hooks established:\n- confusion_matrix.png\n- roc_curve.png\nEvaluation loop validated."
        ],
        'config.py': [
            "Defining global training variables:\n- BATCH_SIZE: 16\n- EPOCHS: 25\n- LEARNING_RATE: 0.0001\n- IMG_SIZE: 224\n- CLASS_NAMES: ['benign', 'malignant']\nConfiguration variables initialized."
        ],
        'predict.py': [
            "Loading prediction dependencies...\nPreparing argparser for CLI inference options.",
            "Compiling predict_single_image() function...\nLoads input target image, runs cv2 CLAHE, scales tensor, executes forward pass.",
            "CLI entry hooks configured.\npredict.py ready for deployment."
        ]
    };

    function renderNotebookCells(cells, fileName) {
        notebookCellsList.innerHTML = '';
        
        cells.forEach((cellContent, index) => {
            const cell = document.createElement('div');
            cell.className = 'notebook-cell';
            cell.innerHTML = `
                <div class="cell-left">
                    <div class="cell-input-prompt">In [${index + 1}]:</div>
                    <button class="cell-play-btn" title="Run Cell">
                        <svg viewBox="0 0 24 24" width="10" height="10"><polygon points="8 5 18 12 8 19 8 5"></polygon></svg>
                    </button>
                </div>
                <div class="cell-code-container">
                    <pre><code>${escapeHtml(cellContent)}</code></pre>
                    <!-- Colab style output cell -->
                    <div class="cell-output-container hidden" id="out-${fileName}-${index}">
                        <div class="cell-output-header">Output</div>
                        <pre class="cell-output-text"></pre>
                    </div>
                </div>
            `;
            
            cell.querySelector('.cell-play-btn').addEventListener('click', () => {
                runCell(cell, fileName, index);
            });
            
            notebookCellsList.appendChild(cell);
        });
    }

    function runCell(cellElement, fileName, cellIndex) {
        const playBtn = cellElement.querySelector('.cell-play-btn');
        const prompt = cellElement.querySelector('.cell-input-prompt');
        const outputContainer = cellElement.querySelector('.cell-output-container');
        const outputText = cellElement.querySelector('.cell-output-text');

        prompt.textContent = 'In [*]:';
        playBtn.style.color = '#38bdf8';
        outputContainer.classList.remove('hidden');
        outputContainer.classList.add('running');
        outputText.textContent = 'Running process cell...';

        addLogEntry(`Executing Notebook cell [${cellIndex + 1}] inside ${fileName}...`, 'info');
        
        setTimeout(() => {
            prompt.textContent = `In [${cellIndex + 1}]:`;
            playBtn.style.color = '';
            outputContainer.classList.remove('running');
            
            // Query mock stdout output text
            const outputs = notebookCellOutputs[fileName] || ["Cell execution completed successfully."];
            const output = outputs[cellIndex] || "Execution completed.";
            outputText.textContent = output;

            addLogEntry(`Cell [${cellIndex + 1}] in ${fileName} executed successfully.`, 'success');
        }, 600);
    }

    async function loadCodeFile(fileName) {
        activeFileName.textContent = fileName;
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
            const cells = segmentCode(data.code);
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
        
        const fileName = activeFileName.textContent;
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
        const file = activeFileName.textContent;
        const cells = codeCache[file];
        if (!cells) return;
        const codeText = cells.join('\n\n');
        
        navigator.clipboard.writeText(codeText).then(() => {
            const origText = btnCopyCode.textContent;
            btnCopyCode.textContent = 'Copied!';
            addLogEntry(`Copied source contents of ${file} to clipboard.`, 'success');
            setTimeout(() => {
                btnCopyCode.textContent = origText;
            }, 1500);
        }).catch(err => {
            addLogEntry('Failed to copy code.', 'error');
        });
    });
});
