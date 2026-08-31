/* ==========================================================================
   OncoVision AI | MSc Dissertation Clinical UI Controller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Navigation Elements
    const btnUiMode = document.getElementById('btnUiMode');
    const btnNoiseMode = document.getElementById('btnNoiseMode');
    const btnDatasetMode = document.getElementById('btnDatasetMode');
    const btnThesisMode = document.getElementById('btnThesisMode');

    const viewWorkstation = document.getElementById('viewWorkstation');
    const viewNoiseLab = document.getElementById('viewNoiseLab');
    const viewDataset = document.getElementById('viewDataset');
    const viewThesis = document.getElementById('viewThesis');

    // Controls
    const loadedModel = document.getElementById('loadedModel');
    const selectModelBackbone = document.getElementById('selectModelBackbone');
    const selectCropStrategy = document.getElementById('selectCropStrategy');
    const chkUseClahe = document.getElementById('chkUseClahe');

    // Upload & Workstation Viewport
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const btnBrowse = document.getElementById('btnBrowse');
    const btnLoadSample = document.getElementById('btnLoadSample');
    const btnChangeImage = document.getElementById('btnChangeImage');
    const viewportArea = document.getElementById('viewportArea');
    const imgOriginal = document.getElementById('imgOriginal');
    const imgSpotlight = document.getElementById('imgSpotlight');
    const systemStatus = document.getElementById('systemStatus');

    // Diagnostic Results
    const verdictBanner = document.getElementById('verdictBanner');
    const txtVerdict = document.getElementById('txtVerdict');
    const txtConfidence = document.getElementById('txtConfidence');
    const txtBenignProb = document.getElementById('txtBenignProb');
    const txtMalignantProb = document.getElementById('txtMalignantProb');
    const barBenign = document.getElementById('barBenign');
    const barMalignant = document.getElementById('barMalignant');

    // Report Items
    const txtModelName = document.getElementById('txtModelName');
    const txtAcousticShadow = document.getElementById('txtAcousticShadow');
    const txtRationale = document.getElementById('txtRationale');

    // Noise Lab Controls
    const selectNoiseType = document.getElementById('selectNoiseType');
    const rangeIntensity = document.getElementById('rangeIntensity');
    const lblIntensityVal = document.getElementById('lblIntensityVal');
    const btnSimulateNoise = document.getElementById('btnSimulateNoise');
    const imgCleanView = document.getElementById('imgCleanView');
    const imgNoisyView = document.getElementById('imgNoisyView');
    const txtCleanResult = document.getElementById('txtCleanResult');
    const txtNoisyResult = document.getElementById('txtNoisyResult');
    const txtNoiseExplanation = document.getElementById('txtNoiseExplanation');

    // Dataset Explorer
    const selectDataset = document.getElementById('selectDataset');
    const selectSplit = document.getElementById('selectSplit');
    const datasetGrid = document.getElementById('datasetGrid');

    // Thesis Reader
    const docList = document.getElementById('docList');
    const markdownViewer = document.getElementById('markdownViewer');

    let currentFile = null;
    let currentGt = null;
    let currentFilename = null;
    let activeInferenceController = null;
    let inferenceVersion = 0;

    function setStatus(message, state = 'ready') {
        systemStatus.textContent = message;
        systemStatus.dataset.state = state;
    }

    async function checkModelAvailability() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            if (health.model_ready) {
                loadedModel.textContent = health.model || 'Trained checkpoint';
                loadedModel.dataset.state = 'ready';
                (health.models || []).forEach(({ name, ready }) => {
                    const option = selectModelBackbone?.querySelector(`option[value="${name}"]`);
                    if (option) option.disabled = !ready;
                });
                if (health.model) selectModelBackbone.value = health.model;
                setStatus('Ready', 'ready');
                return;
            }
            loadedModel.textContent = 'Checkpoint unavailable';
            loadedModel.dataset.state = 'error';
            setStatus('Model unavailable', 'error');
        } catch {
            loadedModel.textContent = 'Server unavailable';
            loadedModel.dataset.state = 'error';
            setStatus('Server unavailable', 'error');
        }
    }

    checkModelAvailability();

    // --- Tab Navigation Handler ---
    function switchTab(activeBtn, activeView) {
        [btnUiMode, btnNoiseMode, btnDatasetMode, btnThesisMode].forEach(b => b?.classList.remove('active'));
        [viewWorkstation, viewNoiseLab, viewDataset, viewThesis].forEach(v => v?.classList.remove('active'));

        activeBtn.classList.add('active');
        activeView.classList.add('active');

        if (activeView === viewDataset) {
            loadDatasetGrid();
        } else if (activeView === viewThesis && !markdownViewer.dataset.loaded) {
            loadThesisDocument('IMAGE_RESIZING_AND_CROPPING_NOISE_ANALYSIS.md');
        }
    }

    btnUiMode?.addEventListener('click', () => switchTab(btnUiMode, viewWorkstation));
    btnNoiseMode?.addEventListener('click', () => switchTab(btnNoiseMode, viewNoiseLab));
    btnDatasetMode?.addEventListener('click', () => switchTab(btnDatasetMode, viewDataset));
    btnThesisMode?.addEventListener('click', () => switchTab(btnThesisMode, viewThesis));

    // --- File Upload & Dropzone Handlers ---
    dropzone?.addEventListener('click', (e) => {
        if (!e.target.closest('button')) {
            fileInput.click();
        }
    });

    btnBrowse?.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    btnLoadSample?.addEventListener('click', (e) => {
        e.stopPropagation();
        loadSampleScan();
    });

    btnChangeImage?.addEventListener('click', () => {
        activeInferenceController?.abort();
        viewportArea.classList.add('hidden');
        dropzone.classList.remove('hidden');
        setStatus('Ready', 'ready');
    });

    fileInput?.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropzone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('hover');
    });

    dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('hover'));

    dropzone?.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('hover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // --- Dynamic Re-run on Setting Change ---
    selectModelBackbone?.addEventListener('change', () => { if (currentFile) runDiagnosticInference(currentFile); });
    selectCropStrategy?.addEventListener('change', () => { if (currentFile) runDiagnosticInference(currentFile); });
    chkUseClahe?.addEventListener('change', () => { if (currentFile) runDiagnosticInference(currentFile); });

    function handleFileUpload(file) {
        const allowedTypes = ['image/png', 'image/jpeg', 'image/tiff'];
        const allowedExtensions = /\.(png|jpe?g|tiff?)$/i;
        if ((!allowedTypes.includes(file.type) && !allowedExtensions.test(file.name)) || file.size > 20 * 1024 * 1024) {
            setStatus('Use a PNG, JPG, or TIFF under 20 MB', 'error');
            return;
        }
        currentFile = file;
        currentGt = null;
        currentFilename = file.name;
        runDiagnosticInference(file);
    }

    async function loadSampleScan(dataset = 'busi') {
        try {
            const res = await fetch(`/api/dataset/files?dataset=${dataset}&split=val`);
            if (res.ok) {
                const files = await res.json();
                if (files && files.length > 0) {
                    const item = files[0];
                    currentFile = item.url;
                    currentGt = item.class.toUpperCase();
                    currentFilename = item.name;
                    runDiagnosticInference(item.url, currentGt, currentFilename);
                    return item;
                }
            }
        } catch (err) {
            console.error('Failed to load sample scan for dataset:', dataset, err);
        }
        const fallbackUrl = `/api/dataset/file/val/malignant/sample_0.png?dataset=${dataset}`;
        currentFile = fallbackUrl;
        currentGt = 'MALIGNANT';
        currentFilename = 'sample_0.png';
        runDiagnosticInference(fallbackUrl, currentGt, currentFilename);
        return { url: fallbackUrl, class: 'MALIGNANT', name: 'sample_0.png' };
    }

    // --- Diagnostic Inference Function ---
    async function runDiagnosticInference(fileOrUrl, explicitGt = null, explicitFilename = null) {
        const requestVersion = ++inferenceVersion;
        activeInferenceController?.abort();
        activeInferenceController = new AbortController();
        setStatus('Processing…', 'loading');

        if (explicitGt) currentGt = explicitGt;
        if (explicitFilename) currentFilename = explicitFilename;

        const formData = new FormData();
        if (fileOrUrl instanceof File) {
            formData.append('file', fileOrUrl);
        } else if (typeof fileOrUrl === 'string') {
            try {
                const res = await fetch(fileOrUrl);
                const blob = await res.blob();
                const fname = currentFilename || (fileOrUrl.split('/').pop().split('?')[0]);
                formData.append('file', blob, fname);
            } catch (err) {
                console.error('Failed to fetch sample image url:', err);
                systemStatus.textContent = 'Fetch Error';
                return;
            }
        }

        if (currentGt) {
            formData.append('ground_truth', currentGt);
        }

        formData.append('use_clahe', chkUseClahe.checked);
        formData.append('crop_strategy', selectCropStrategy.value);
        formData.append('model_name', selectModelBackbone.value);
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
                signal: activeInferenceController.signal
            });

            if (!response.ok) {
                const detail = await response.json().catch(() => ({}));
                const msg = detail.detail || 'Inference failed';
                if (msg.includes('NON_BREAST_ULTRASOUND')) {
                    displayRejectionResult(msg);
                    setStatus('Non-Ultrasound Study Rejected', 'error');
                } else {
                    setStatus(msg, 'error');
                }
                return;
            }

            const data = await response.json();
            if (requestVersion !== inferenceVersion) return;
            if (explicitGt) {
                data.ground_truth = explicitGt;
            }
            displayDiagnosticResults(data);
            setStatus('Complete', 'success');

        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error('Inference error:', err);
                setStatus('Server error: please try again', 'error');
            }
        }
    }

    // --- Display Diagnostic Results ---
    function displayDiagnosticResults(data) {
        viewportArea.classList.remove('hidden');
        dropzone.classList.add('hidden');
        btnChangeImage.classList.remove('hidden');

        document.getElementById('phOriginal')?.classList.add('hidden');
        document.getElementById('phSpotlight')?.classList.add('hidden');

        imgOriginal.src = data.original_image;
        imgOriginal.alt = `Original ultrasound scan: ${data.filename || 'uploaded image'}`;
        imgOriginal.classList.remove('hidden');

        imgSpotlight.src = data.processed_image;
        imgSpotlight.alt = `Research visualisation with localized region overlay: ${data.prediction}`;
        imgSpotlight.classList.remove('hidden');

        // Ground Truth Tag Badge
        const gtBadge = document.getElementById('gtTagBadge');
        const gt = (data.ground_truth || 'UNKNOWN').toUpperCase();
        if (gtBadge) {
            gtBadge.classList.remove('hidden');
            gtBadge.textContent = `GT Tag: ${gt}`;
            gtBadge.className = 'gt-tag-badge ' + (gt === 'BENIGN' ? 'benign' : (gt === 'MALIGNANT' ? 'malignant' : 'unknown'));
        }

        // Verdict Banner
        const pred = data.prediction.toUpperCase();
        txtVerdict.textContent = pred;
        txtConfidence.textContent = `${data.confidence}%`;

        verdictBanner.className = 'verdict-banner ' + (pred === 'BENIGN' ? 'benign' : 'malignant');

        // Probability Bars
        const bProb = data.probabilities.benign || 0;
        const mProb = data.probabilities.malignant || 0;

        txtBenignProb.textContent = `${bProb}%`;
        txtMalignantProb.textContent = `${mProb}%`;
        barBenign.style.transform = `scaleX(${bProb / 100})`;
        barMalignant.style.transform = `scaleX(${mProb / 100})`;

        txtModelName.textContent = data.model_name || '--';
        txtAcousticShadow.textContent = data.noise_analysis?.dominant_type || '--';
        txtRationale.textContent = data.research_notice || 'Experimental research output only. Not for clinical use.';
    }

    function displayRejectionResult(message) {
        viewportArea.classList.remove('hidden');
        dropzone.classList.add('hidden');
        btnChangeImage.classList.remove('hidden');

        const gtBadge = document.getElementById('gtTagBadge');
        if (gtBadge) {
            gtBadge.classList.remove('hidden');
            gtBadge.textContent = 'REJECTED: NON-ULTRASOUND';
            gtBadge.className = 'gt-tag-badge unknown';
        }

        txtVerdict.textContent = 'NON-BREAST ULTRASOUND';
        txtConfidence.textContent = '0.0%';
        verdictBanner.className = 'verdict-banner unknown';

        txtBenignProb.textContent = '0.0%';
        txtMalignantProb.textContent = '0.0%';
        barBenign.style.transform = 'scaleX(0)';
        barMalignant.style.transform = 'scaleX(0)';

        txtModelName.textContent = 'Domain Validation Engine';
        txtAcousticShadow.textContent = 'Non-Medical Study';
        txtRationale.textContent = '⚠️ ' + message.replace('NON_BREAST_ULTRASOUND: ', '');

        // Pop up the Rejection Modal
        openRejectionModal(message);
    }

    // --- NON-BREAST ULTRASOUND REJECTION MODAL CONTROLLER ---
    const modalRejection = document.getElementById('modalRejection');
    const modalRejectionMessage = document.getElementById('modalRejectionMessage');
    const btnModalClose = document.getElementById('btnModalClose');
    const btnModalDismiss = document.getElementById('btnModalDismiss');

    function openRejectionModal(rawMessage) {
        if (!modalRejection) return;
        const cleanMsg = rawMessage.replace('NON_BREAST_ULTRASOUND: ', '');
        if (modalRejectionMessage) {
            modalRejectionMessage.textContent = cleanMsg;
        }
        modalRejection.classList.remove('hidden');
    }

    function closeRejectionModal() {
        if (modalRejection) {
            modalRejection.classList.add('hidden');
        }
        activeInferenceController?.abort();
        viewportArea.classList.add('hidden');
        dropzone.classList.remove('hidden');
        setStatus('Ready', 'ready');
    }

    btnModalClose?.addEventListener('click', closeRejectionModal);
    btnModalDismiss?.addEventListener('click', () => {
        closeRejectionModal();
        fileInput?.click();
    });
    modalRejection?.addEventListener('click', (e) => {
        if (e.target === modalRejection) closeRejectionModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modalRejection && !modalRejection.classList.contains('hidden')) {
            closeRejectionModal();
        }
    });

    // --- Noise Laboratory 4-Column Suite Handler ---
    const selectNoiseDataset = document.getElementById('selectNoiseDataset');
    const btnNoiseLoadSample = document.getElementById('btnNoiseLoadSample');

    const selectCol3Strategy = document.getElementById('selectCol3Strategy');
    const selectCol4Strategy = document.getElementById('selectCol4Strategy');

    const imgCol1Clean = document.getElementById('imgCol1Clean');
    const txtCol1Pred = document.getElementById('txtCol1Pred');
    const txtCol1Conf = document.getElementById('txtCol1Conf');
    const cardCol1Verdict = document.getElementById('cardCol1Verdict');

    const imgCol2Noisy = document.getElementById('imgCol2Noisy');
    const txtCol2Pred = document.getElementById('txtCol2Pred');
    const txtCol2Conf = document.getElementById('txtCol2Conf');
    const cardCol2Verdict = document.getElementById('cardCol2Verdict');

    const imgCol3Direct = document.getElementById('imgCol3Direct');
    const txtCol3Pred = document.getElementById('txtCol3Pred');
    const txtCol3Conf = document.getElementById('txtCol3Conf');
    const cardCol3Verdict = document.getElementById('cardCol3Verdict');

    const imgCol4Roi = document.getElementById('imgCol4Roi');
    const txtCol4Pred = document.getElementById('txtCol4Pred');
    const txtCol4Conf = document.getElementById('txtCol4Conf');
    const cardCol4Verdict = document.getElementById('cardCol4Verdict');

    rangeIntensity?.addEventListener('input', (e) => {
        lblIntensityVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    selectNoiseDataset?.addEventListener('change', async () => {
        const dataset = selectNoiseDataset.value;
        await loadSampleScan(dataset);
        setTimeout(runNoiseAnalysisSuite, 350);
    });

    btnNoiseLoadSample?.addEventListener('click', async () => {
        const dataset = selectNoiseDataset.value;
        await loadSampleScan(dataset);
        setTimeout(runNoiseAnalysisSuite, 350);
    });

    selectCol3Strategy?.addEventListener('change', runNoiseAnalysisSuite);
    selectNoiseType?.addEventListener('change', runNoiseAnalysisSuite);

    async function runNoiseAnalysisSuite() {
        if (!currentFile) {
            const dataset = selectNoiseDataset.value || 'busi';
            await loadSampleScan(dataset);
        }

        const formData = new FormData();
        if (currentFile instanceof File) {
            formData.append('file', currentFile);
        } else {
            const res = await fetch(currentFile);
            const blob = await res.blob();
            formData.append('file', blob, 'sample.png');
        }

        formData.append('noise_type', selectNoiseType.value);
        formData.append('intensity', rangeIntensity.value);
        formData.append('use_clahe', chkUseClahe.checked);
        if (selectCol3Strategy) formData.append('col3_strategy', selectCol3Strategy.value);
        formData.append('model_name', selectModelBackbone.value);
        formData.append('seed', '42');

        try {
            const response = await fetch('/api/noise/simulate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                alert('Noise simulation failed.');
                return;
            }

            const resData = await response.json();

            // Column 1: Clean
            if (resData.stage1_clean) {
                document.getElementById('phCol1')?.classList.add('hidden');
                imgCol1Clean.src = resData.stage1_clean.image;
                imgCol1Clean.classList.remove('hidden');
                txtCol1Pred.textContent = resData.stage1_clean.prediction;
                txtCol1Conf.textContent = `${resData.stage1_clean.confidence}%`;
                cardCol1Verdict.className = 'verdict-mini-card ' + (resData.stage1_clean.prediction === 'BENIGN' ? 'success' : 'malignant');
            }

            // Column 2: Noisy Uncropped
            if (resData.stage2_noisy) {
                document.getElementById('phCol2')?.classList.add('hidden');
                imgCol2Noisy.src = resData.stage2_noisy.image;
                imgCol2Noisy.classList.remove('hidden');
                txtCol2Pred.textContent = resData.stage2_noisy.prediction;
                txtCol2Conf.textContent = `${resData.stage2_noisy.confidence}%`;
                cardCol2Verdict.className = 'verdict-mini-card ' + (resData.stage2_noisy.prediction === 'BENIGN' ? 'success' : 'malignant');
            }

            // Column 3: Strategy A
            if (resData.stage3_direct) {
                document.getElementById('phCol3')?.classList.add('hidden');
                imgCol3Direct.src = resData.stage3_direct.image;
                imgCol3Direct.classList.remove('hidden');
                txtCol3Pred.textContent = resData.stage3_direct.prediction;
                txtCol3Conf.textContent = `${resData.stage3_direct.confidence}%`;
                cardCol3Verdict.className = 'verdict-mini-card ' + (resData.stage3_direct.prediction === 'BENIGN' ? 'success' : 'warning');
            }

            // Column 4: Strategy B
            if (resData.stage4_roi) {
                document.getElementById('phCol4')?.classList.add('hidden');
                imgCol4Roi.src = resData.stage4_roi.image;
                imgCol4Roi.classList.remove('hidden');
                txtCol4Pred.textContent = resData.stage4_roi.prediction;
                txtCol4Conf.textContent = `${resData.stage4_roi.confidence}%`;
                cardCol4Verdict.className = 'verdict-mini-card ' + (resData.stage4_roi.prediction === 'BENIGN' ? 'success' : 'malignant');
            }

            const delta = resData.deltas || {};
            txtNoiseExplanation.textContent = `${resData.explanation} Same noise realization (seed ${resData.seed}) · noise vs clean: ${delta.noise_vs_clean >= 0 ? '+' : ''}${delta.noise_vs_clean ?? '--'} pts · method A vs noise: ${delta.method_a_vs_noise >= 0 ? '+' : ''}${delta.method_a_vs_noise ?? '--'} pts · proposed ROI vs noise: ${delta.proposed_vs_noise >= 0 ? '+' : ''}${delta.proposed_vs_noise ?? '--'} pts. These are observed model outputs, not adjusted percentages.`;

        } catch (err) {
            console.error('Noise simulation error:', err);
        }
    }

    btnSimulateNoise?.addEventListener('click', runNoiseAnalysisSuite);

    // --- Dataset Explorer Loader ---
    async function loadDatasetGrid() {
        const dataset = selectDataset.value;
        const split = selectSplit.value;

        datasetGrid.innerHTML = '<p class="placeholder-text">Loading dataset items...</p>';

        try {
            const res = await fetch(`/api/dataset/files?dataset=${dataset}&split=${split}`);
            if (!res.ok) return;

            const files = await res.json();
            datasetGrid.innerHTML = '';

            if (!files.length) {
                datasetGrid.innerHTML = '<p class="placeholder-text">No images are available in this dataset partition.</p>';
                return;
            }
            files.slice(0, 30).forEach(item => {
                const card = document.createElement('button');
                card.type = 'button';
                card.className = 'dataset-card';
                const image = document.createElement('img');
                image.src = item.url;
                image.alt = `Dataset scan: ${item.name}`;
                image.loading = 'lazy';
                const name = document.createElement('span');
                name.className = 'd-name';
                name.textContent = item.name;
                const classification = document.createElement('span');
                classification.className = `d-class ${item.class}`;
                classification.textContent = item.class;
                card.append(image, name, classification);
                card.addEventListener('click', () => {
                    switchTab(btnUiMode, viewWorkstation);
                    currentFile = item.url;
                    runDiagnosticInference(item.url, item.class.toUpperCase(), item.name);
                });
                datasetGrid.appendChild(card);
            });
        } catch (err) {
            console.error('Failed to load dataset files:', err);
            datasetGrid.innerHTML = '<p class="placeholder-text">Dataset items could not be loaded. Try again when the server is available.</p>';
        }
    }

    selectDataset?.addEventListener('change', loadDatasetGrid);
    selectSplit?.addEventListener('change', loadDatasetGrid);

    // --- Thesis Markdown Reader ---
    docList?.querySelectorAll('.doc-item').forEach(item => {
        item.addEventListener('click', () => {
            docList.querySelectorAll('.doc-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            loadThesisDocument(item.dataset.file);
        });
    });

    async function loadThesisDocument(filename) {
        markdownViewer.innerHTML = '<p class="placeholder-text">Loading document...</p>';
        try {
            const response = await fetch(`/api/code/${filename}`);
            if (!response.ok) {
                markdownViewer.innerHTML = '<p class="placeholder-text">Document not found.</p>';
                return;
            }
            const data = await response.json();
            if (window.marked) {
                markdownViewer.innerHTML = window.marked.parse(data.code);
                if (window.renderMathInElement) {
                    window.renderMathInElement(markdownViewer, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false},
                            {left: '\\(', right: '\\)', display: false},
                            {left: '\\[', right: '\\]', display: true}
                        ]
                    });
                }
            } else {
                markdownViewer.innerHTML = `<pre>${data.code}</pre>`;
            }
            markdownViewer.dataset.loaded = "true";
        } catch (err) {
            console.error('Error loading thesis doc:', err);
            markdownViewer.innerHTML = '<p class="placeholder-text">Error loading document.</p>';
        }
    }
});
