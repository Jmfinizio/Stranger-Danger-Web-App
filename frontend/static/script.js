// ==================== Constants & Global State ====================
const API_BASE_URL = window.location.origin; // Base URL for API calls, using current origin

let currentFile = null; // Holds the currently selected video file (Blob or File)
let analysisAbortController = null; // Controller to abort video analysis requests
let spCredentials = {}; // Stores SharePoint credentials after connection
let isSharePointFile = false; // Flag indicating if the file came from SharePoint
let progressSource = null; // EventSource for server-sent events during processing

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    initApp(); // Kick off app setup
});

function initApp() {
    initEventListeners(); // Attach all UI event handlers
    showScreen('initialScreen'); // Display the upload/timestamp screen
}

function initEventListeners() {
    // File upload via click
    document.getElementById('dropZone').addEventListener('click', () => {
        document.getElementById('videoInput').click(); // Trigger hidden file input
    });

    // File input change handler
    document.getElementById('videoInput').addEventListener('change', handleFileSelect);

    // Drag-and-drop handlers
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', handleDragOver); // Highlight zone on drag over
    dropZone.addEventListener('drop', handleDrop);        // Handle file drop
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover')); // Remove highlight

    // Navigation buttons to switch screens
    document.querySelectorAll('[data-screen]').forEach(btn => {
        btn.addEventListener('click', () => {
            if (analysisAbortController) {
                // If analysis in flight, cancel then switch
                cancelAnalysis().finally(() => showScreen(btn.dataset.screen));
            } else {
                showScreen(btn.dataset.screen);
            }
        });
    });

    // "New Analysis" button on results screen
    document.querySelector('#resultsScreen .btn.secondary').addEventListener('click', handleNewAnalysis);

    // Analysis control buttons
    document.getElementById('analyzeBtn').addEventListener('click', startAnalysis); // Start processing
    document.getElementById('cancelBtn').addEventListener('click', cancelAnalysis); // Cancel processing
    document.getElementById('downloadBtn').addEventListener('click', () => {
        // Download handled dynamically in setupDownload()
    });

    // Timestamp input validation handlers
    document.getElementById('timestamp1').addEventListener('input', validateTimestamps);
    document.getElementById('timestamp2').addEventListener('input', validateTimestamps);
    document.getElementById('timestamp3').addEventListener('input', validateTimestamps);
}

// ==================== High-Level Workflows ====================

// Start a brand new analysis (from results screen)
async function handleNewAnalysis() {
    try {
        await cancelAnalysis(); // Abort any running job
        resetApp();            // Clear form and state
        resetAnalyzeButton();  // Restore Analyze button
        showScreen('initialScreen'); // Go back to upload
    } catch (error) {
        showError(`Failed to start new analysis: ${error.message}`); // Show error
    }
}

// Handle SharePoint credentials submission and file listing
async function handleSpCredSubmit(event) {
    event.preventDefault();

    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true; // Prevent double submits
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...'; // Show spinner

    // Collect credentials from form
    spCredentials = {
        siteUrl: document.getElementById('spSiteUrl').value.trim(),
        clientId: document.getElementById('spClientId').value.trim(),
        clientSecret: document.getElementById('spClientSecret').value.trim(),
        docLibrary: document.getElementById('spDocLibrary').value.trim()
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/sharepoint/files`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                ...spCredentials,
                doc_library: spCredentials.docLibrary
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || response.statusText);
        }

        const files = await response.json(); // Array of SharePoint files
        renderSpFileList(files);             // Populate file list UI
        showScreen('sharepointFileScreen');  // Switch to file selection
    } catch (error) {
        showError(`SharePoint connection failed: ${error.message}`);
    } finally {
        submitBtn.disabled = false;       // Restore button
        submitBtn.innerHTML = originalText;
    }
}

// Render list of SharePoint files with Select buttons
function renderSpFileList(files) {
    const fileList = document.getElementById('spFileList');
    if (!fileList) return;

    fileList.innerHTML = files.map(file => `
        <div class="sp-file-item">
            <span>${file.name}</span>
            <button class="btn" onclick="handleSpFile('${file.id}')">
                <i class="fas fa-play"></i> Select
            </button>
        </div>
    `).join('');
}

// Handle selecting and downloading a file from SharePoint
async function handleSpFile(fileId) {
    const selectBtn = event.target;
    const originalText = selectBtn.innerHTML;
    selectBtn.disabled = true;
    selectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';

    try {
        const formData = new URLSearchParams({
            ...spCredentials,
            file_id: fileId
        });

        const response = await fetch(`${API_BASE_URL}/api/sharepoint/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || response.statusText);
        }

        currentFile = await response.blob(); // Store the downloaded blob
        isSharePointFile = true;             // Mark as SharePoint source
        await startAnalysis();               // Begin processing
    } catch (error) {
        showError(`File download failed: ${error.message}`);
    } finally {
        selectBtn.disabled = false;           // Restore button
        selectBtn.innerHTML = originalText;
    }
}

// Kick off video analysis by sending file and timestamps to backend
async function startAnalysis() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true; // Prevent re-click
    analyzeBtn.onclick = null;
    analyzeBtn.innerText = 'Analyzing…'; // Update label

    if (!currentFile) {
        showError('Please select a file first!');
        resetAnalyzeButton();
        return;
    }

    const t1 = document.getElementById('timestamp1').value;
    const t2 = document.getElementById('timestamp2').value;
    const t3 = document.getElementById('timestamp3').value;
    if (!validateTimeOrder(t1, t2, t3)) {
        showError('Timestamps must be in ascending order');
        resetAnalyzeButton();
        return;
    }

    showScreen('progressScreen'); // Show progress UI
    analysisAbortController = new AbortController(); // New controller

    try {
        const formData = new FormData();
        formData.append('video', currentFile);
        formData.append('timestamp1', t1);
        formData.append('timestamp2', t2);
        formData.append('timestamp3', t3);

        const response = await fetch(`${API_BASE_URL}/api/process-video`, {
            method: 'POST',
            body: formData,
            signal: analysisAbortController.signal
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || response.statusText);
        }

        const { process_id } = await response.json();
        setupProgressTracker(process_id);

    } catch (error) {
        if (error.name !== 'AbortError') {
            showError(`Analysis failed: ${error.message}`);
            showScreen('initialScreen');
        }
    }
}

// ==================== File Upload/Selection Handlers ====================

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) handleFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) {
        showError('Please upload a valid video file (MP4, MOV, or AVI)');
        return;
    }

    currentFile = file;
    isSharePointFile = false;

    const preview = document.getElementById('videoPreview');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (preview.src) URL.revokeObjectURL(preview.src);

    preview.src = URL.createObjectURL(file);
    preview.classList.remove('hidden');
    analyzeBtn.disabled = false;

    document.getElementById('timestamp1').value = '';
    document.getElementById('timestamp2').value = '';
    document.getElementById('timestamp3').value = '';
    validateTimestamps();
}

function showUploadScreen(type) {
    if (type === 'sharepoint') {
        showScreen('sharepointCredScreen');
    } else {
        showScreen('localUploadScreen');
    }
}

// ==================== Progress Tracking ====================

function setupProgressTracker(processId) {
    if (progressSource) progressSource.close();

    progressSource = new EventSource(`${API_BASE_URL}/api/progress/${processId}`);

    progressSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.status === 'completed') {
                handleAnalysisComplete(processId);
                progressSource.close();
            } else if (data.status === 'error') {
                showError(data.error || 'Analysis failed');
                progressSource.close();
                showScreen('initialScreen');
            } else {
                updateProgressUI(data);
            }
        } catch (error) {
            console.error('Error parsing progress:', error);
        }
    };

    progressSource.onerror = () => {
        console.log('SSE error - attempting reconnect');
        setTimeout(() => setupProgressTracker(processId), 2000);
    };
}

function updateProgressUI(progress) {
    const progressBar = document.getElementById('progressBar');
    const progressMessage = document.getElementById('progressMessage');

    progressBar.style.width = `${progress.percent}%`;
    progressMessage.textContent = progress.message;

    if (progress.current && progress.total) {
        document.getElementById('frameCounter').textContent = `${progress.current}/${progress.total} seconds processed`;
    }
}

async function handleAnalysisComplete(processId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/results/${processId}`);
        const blob = await response.blob();
        setupDownload(blob);
        showScreen('resultsScreen');
    } catch (error) {
        showError('Failed to retrieve results');
    }
}

// ==================== Utilities ====================

function showScreen(screenId) {
    document.querySelectorAll('.card').forEach(el => el.classList.add('hidden'));
    const targetScreen = document.getElementById(screenId);
    if (targetScreen) {
        targetScreen.classList.remove('hidden');
        window.scrollTo(0, 0);
    } else {
        console.error(`Screen with ID ${screenId} not found`);
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i><span>${message}</span>`;
    document.body.prepend(errorDiv);
    setTimeout(() => { errorDiv.classList.add('fade-out'); setTimeout(() => errorDiv.remove(), 500); }, 5000);
}

function validateTimestamps() {
    const t1 = document.getElementById('timestamp1');
    const t2 = document.getElementById('timestamp2');
    const t3 = document.getElementById('timestamp3');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const isValid = t1.checkValidity() && t2.checkValidity() && t3.checkValidity() && t1.value !== '' && t2.value !== '' && t3.value !== '';
    analyzeBtn.disabled = !isValid;
}

function validateTimeOrder(t1, t2, t3) {
    const toSeconds = t => { const [h, m, s] = t.split(':').map(Number); return h*3600 + m*60 + s; };
    return toSeconds(t1) < toSeconds(t2) && toSeconds(t2) < toSeconds(t3);
}

function resetAnalyzeButton() {
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = false;
    btn.innerText = 'Start Analysis';
    btn.onclick = startAnalysis;
}

function resetApp() {
    const preview = document.getElementById('videoPreview');
    if (preview.src) URL.revokeObjectURL(preview.src);
        preview.src = '';
        preview.classList.add('hidden');
        document.getElementById('videoInput').value = '';
        const progressBar = document.getElementById('progressBar'); if (progressBar) progressBar.style.width = '0%';
        const progressMessage = document.getElementById('progressMessage'); if (progressMessage) progressMessage.textContent = '';
        const spForm = document.getElementById('spCredForm'); if (spForm) spForm.reset();
    if (progressSource) { progressSource.close(); progressSource = null; }
        currentFile = null;
        isSharePointFile = false;
        spCredentials = {};
}

async function cancelAnalysis() {
    try {
        if (progressSource) {
            progressSource.close(); 
            progressSource = null;
          }
        if (!analysisAbortController) return;
        const progressMessage = document.getElementById('progressMessage'); if (progressMessage) progressMessage.textContent = "Cancelling analysis...";
        analysisAbortController.abort();
        await fetch(`${API_BASE_URL}/api/cancel-analysis`, { method: 'POST' });
    } catch (error) {
        console.error('Cancellation error:', error);
        throw error;
    } finally {
        analysisAbortController = null;
    }
}

function setupDownload(blob) {
    const url = URL.createObjectURL(blob);
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.onclick = null;
    downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = url;
        a.download = `stranger_danger_analysis_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
    };
}

// ==================== Global Exports ====================

window.showUploadScreen = showUploadScreen;
window.handleSpCredSubmit = handleSpCredSubmit;
window.handleSpFile = handleSpFile;
window.startAnalysis = startAnalysis;
window.cancelAnalysis = cancelAnalysis;
window.resetApp = resetApp;
