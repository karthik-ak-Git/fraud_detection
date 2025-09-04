// Fraud Detection Frontend JavaScript

// API configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM elements
const transactionForm = document.getElementById('transactionForm');
const generateSampleBtn = document.getElementById('generateSampleBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsContainer = document.getElementById('resultsContainer');
const feedbackForm = document.getElementById('feedbackForm');
const feedbackSection = document.getElementById('feedbackSection');
const noFeedbackMessage = document.getElementById('noFeedbackMessage');
const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

// Global variables
let lastPredictionResult = null;
let lastTransactionData = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeForm();
    checkAPIHealth();
    loadStatistics();
    setupEventListeners();
});

// Initialize form elements
function initializeForm() {
    // Populate hour dropdown
    const hourSelect = document.getElementById('hour');
    for (let i = 0; i < 24; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `${i.toString().padStart(2, '0')}:00`;
        hourSelect.appendChild(option);
    }

    // Set default values
    document.getElementById('hour').value = new Date().getHours();
    document.getElementById('day_of_week').value = new Date().getDay();
}

// Setup event listeners
function setupEventListeners() {
    // Transaction form submission
    transactionForm.addEventListener('submit', handleTransactionSubmit);

    // Generate sample transaction
    generateSampleBtn.addEventListener('click', generateSampleTransaction);

    // Feedback form submission
    feedbackForm.addEventListener('submit', handleFeedbackSubmit);

    // Feedback radio button changes
    document.querySelectorAll('input[name="feedback"]').forEach(radio => {
        radio.addEventListener('change', handleFeedbackRadioChange);
    });

    // Auto-update statistics every 30 seconds
    setInterval(loadStatistics, 30000);
}

// Check API health status
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        const statusElement = document.getElementById('modelStatus');
        if (data.status === 'healthy' && data.model_loaded) {
            statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Model Ready';
            statusElement.className = 'navbar-text text-success';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle text-warning"></i> Model Loading';
            statusElement.className = 'navbar-text text-warning';
        }
    } catch (error) {
        console.error('API health check failed:', error);
        const statusElement = document.getElementById('modelStatus');
        statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> API Offline';
        statusElement.className = 'navbar-text text-danger';
    }
}

// Load system statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();

        // Update statistics display
        document.getElementById('totalPredictions').textContent = '-';
        document.getElementById('fraudDetected').textContent = '-';
        document.getElementById('fraudRate').textContent = '-';
        document.getElementById('modelAccuracy').textContent = data.model_accuracy || '-';

        if (data.feedback_count > 0) {
            document.getElementById('totalPredictions').textContent = data.feedback_count;
            document.getElementById('fraudDetected').textContent = data.correct_predictions || 0;
            const fraudRate = ((data.correct_predictions / data.feedback_count) * 100).toFixed(1);
            document.getElementById('fraudRate').textContent = fraudRate;
        }

    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// Handle transaction form submission
async function handleTransactionSubmit(event) {
    event.preventDefault();

    try {
        // Show loading modal with debugging
        console.log('Showing loading modal...');
        loadingModal.show();

        // Ensure modal is visible
        setTimeout(() => {
            const modalElement = document.getElementById('loadingModal');
            if (modalElement && !modalElement.classList.contains('show')) {
                console.warn('Modal not showing properly, forcing display');
                modalElement.style.display = 'block';
                modalElement.classList.add('show');
            }
        }, 50);

        // Collect form data
        const transactionData = collectTransactionData();
        console.log('Transaction data:', transactionData);

        // Make prediction request with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Prediction result received:', result);

        // Store results for feedback
        lastPredictionResult = result;
        lastTransactionData = transactionData;

        // Display results
        displayPredictionResult(result);

        // Enable feedback section
        enableFeedbackSection();

        // Show success toast
        showToast('Success! Transaction analyzed successfully.', 'success');

    } catch (error) {
        console.error('Prediction failed:', error);
        showToast('Failed to analyze transaction. Please try again.', 'error');
        displayError('Failed to analyze transaction. Please check your connection and try again.');
    } finally {
        // Hide loading modal with multiple approaches for reliability
        setTimeout(() => {
            try {
                loadingModal.hide();
                // Force hide if bootstrap modal doesn't work
                const modalElement = document.getElementById('loadingModal');
                if (modalElement) {
                    modalElement.style.display = 'none';
                    modalElement.classList.remove('show');
                    // Remove backdrop if it exists
                    const backdrop = document.querySelector('.modal-backdrop');
                    if (backdrop) {
                        backdrop.remove();
                    }
                    // Restore body classes
                    document.body.classList.remove('modal-open');
                    document.body.style.overflow = '';
                    document.body.style.paddingRight = '';
                }
            } catch (modalError) {
                console.error('Error hiding modal:', modalError);
            }
        }, 100); // Small delay to ensure all operations complete
    }
}

// Collect transaction data from form
function collectTransactionData() {
    const data = {
        amount: parseFloat(document.getElementById('amount').value) || 0,
        oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value) || 0,
        newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value) || 0,
        oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value) || 0,
        newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value) || 0,
        hour: parseInt(document.getElementById('hour').value) || 0,
        day_of_week: parseInt(document.getElementById('day_of_week').value) || 0,
        transaction_count_1h: parseInt(document.getElementById('transaction_count_1h').value) || 1,
        transaction_count_24h: parseInt(document.getElementById('transaction_count_24h').value) || 1,
        type_CASH_IN: 0,
        type_CASH_OUT: 0,
        type_DEBIT: 0,
        type_PAYMENT: 0,
        type_TRANSFER: 0
    };

    // Set transaction type
    const selectedType = document.querySelector('input[name="transactionType"]:checked');
    if (selectedType) {
        data[`type_${selectedType.value}`] = 1;
    }

    return data;
}

// Display prediction result
function displayPredictionResult(result) {
    const isFraud = result.prediction === 'fraud';
    const confidence = result.confidence;
    const riskScore = result.risk_score;
    const fraudProb = (result.fraud_probability * 100).toFixed(1);

    // Format confidence properly
    const confidenceDisplay = typeof confidence === 'string'
        ? confidence.toUpperCase()
        : `${(confidence * 100).toFixed(1)}%`;

    // Determine risk level
    let riskLevel, riskClass;
    if (riskScore >= 7) {
        riskLevel = 'HIGH RISK';
        riskClass = 'risk-high';
    } else if (riskScore >= 4) {
        riskLevel = 'MEDIUM RISK';
        riskClass = 'risk-medium';
    } else {
        riskLevel = 'LOW RISK';
        riskClass = 'risk-low';
    }

    const resultHTML = `
        <div class="result-enter ${isFraud ? 'fraud-result' : 'normal-result'}">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="d-flex align-items-center mb-3">
                        <i class="fas fa-${isFraud ? 'exclamation-triangle' : 'check-circle'} fa-2x me-3"></i>
                        <div>
                            <h3 class="mb-1">${isFraud ? 'POTENTIAL FRAUD DETECTED' : 'TRANSACTION APPEARS NORMAL'}</h3>
                            <p class="mb-0">Confidence: ${confidenceDisplay}</p>
                        </div>
                    </div>
                    
                    <div class="row text-center">
                        <div class="col-6">
                            <div class="border-end border-light">
                                <div class="h5 mb-1">${fraudProb}%</div>
                                <small>Fraud Probability</small>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="h5 mb-1">${(100 - fraudProb)}%</div>
                            <small>Normal Probability</small>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4 text-center">
                    <div class="risk-score ${riskClass}">${riskScore}</div>
                    <div class="h6 mb-0">${riskLevel}</div>
                    <small>Risk Score (0-10)</small>
                </div>
            </div>
            
            <div class="mt-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>Fraud Confidence</span>
                    <span>${fraudProb}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill confidence-${confidence}" 
                         style="width: ${fraudProb}%"></div>
                </div>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-body">
                <h6 class="card-title">
                    <i class="fas fa-info-circle me-2"></i>
                    Analysis Details
                </h6>
                <div class="row">
                    <div class="col-md-6">
                        <ul class="list-unstyled mb-0">
                            <li><strong>Model Version:</strong> ${result.model_version}</li>
                            <li><strong>Analysis Time:</strong> ${new Date(result.timestamp).toLocaleString()}</li>
                            <li><strong>Processing:</strong> Real-time AI analysis</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <div class="alert alert-${isFraud ? 'danger' : 'success'} mb-0 py-2">
                            <i class="fas fa-${isFraud ? 'shield-alt' : 'thumbs-up'} me-2"></i>
                            ${isFraud ?
            'Consider additional verification steps for this transaction.' :
            'This transaction shows normal patterns and low risk indicators.'
        }
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    resultsContainer.innerHTML = resultHTML;
}

// Display error message
function displayError(message) {
    const errorHTML = `
        <div class="alert alert-danger text-center">
            <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
            <h5>Analysis Failed</h5>
            <p class="mb-0">${message}</p>
        </div>
    `;
    resultsContainer.innerHTML = errorHTML;
}

// Generate sample transaction data
function generateSampleTransaction() {
    // Generate realistic sample data
    const sampleTypes = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'];
    const selectedType = sampleTypes[Math.floor(Math.random() * sampleTypes.length)];

    const amount = Math.random() * 5000 + 100; // $100 - $5100
    const oldBalance = Math.random() * 10000 + 1000; // $1000 - $11000
    const newBalance = Math.max(0, oldBalance - amount + (Math.random() * 1000 - 500));

    // Fill form with sample data
    document.getElementById('amount').value = amount.toFixed(2);
    document.getElementById('oldbalanceOrg').value = oldBalance.toFixed(2);
    document.getElementById('newbalanceOrig').value = newBalance.toFixed(2);
    document.getElementById('oldbalanceDest').value = (Math.random() * 5000).toFixed(2);
    document.getElementById('newbalanceDest').value = (Math.random() * 5000 + amount).toFixed(2);
    document.getElementById('hour').value = Math.floor(Math.random() * 24);
    document.getElementById('day_of_week').value = Math.floor(Math.random() * 7);
    document.getElementById('transaction_count_1h').value = Math.floor(Math.random() * 5) + 1;
    document.getElementById('transaction_count_24h').value = Math.floor(Math.random() * 20) + 5;

    // Select transaction type
    document.querySelector(`input[value="${selectedType}"]`).checked = true;

    showToast('Sample transaction generated successfully!', 'success');
}

// Enable feedback section
function enableFeedbackSection() {
    feedbackSection.style.display = 'block';
    noFeedbackMessage.style.display = 'none';

    // Store prediction data in hidden fields
    document.getElementById('lastPrediction').value = JSON.stringify(lastPredictionResult);
    document.getElementById('lastTransactionData').value = JSON.stringify(lastTransactionData);
}

// Handle feedback radio button changes
function handleFeedbackRadioChange(event) {
    const correctClassSection = document.getElementById('correctClassSection');
    if (event.target.value === 'incorrect') {
        correctClassSection.style.display = 'block';
        document.getElementById('correctClass').required = true;
    } else {
        correctClassSection.style.display = 'none';
        document.getElementById('correctClass').required = false;
    }
}

// Handle feedback form submission
async function handleFeedbackSubmit(event) {
    event.preventDefault();

    try {
        const feedbackType = document.querySelector('input[name="feedback"]:checked').value;
        let actualClass;

        if (feedbackType === 'correct') {
            actualClass = lastPredictionResult.prediction;
        } else {
            actualClass = document.getElementById('correctClass').value;
        }

        const feedbackData = {
            predicted_class: lastPredictionResult.prediction,
            actual_class: actualClass,
            transaction_data: lastTransactionData,
            user_comment: document.getElementById('feedbackComment').value,
            transaction_id: `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };

        const response = await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedbackData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Show success message
        showToast('Thank you for your feedback! This helps improve our model.', 'success');

        // Reset feedback form
        feedbackForm.reset();
        feedbackSection.style.display = 'none';
        noFeedbackMessage.style.display = 'block';
        document.getElementById('correctClassSection').style.display = 'none';

        // Reload statistics
        loadStatistics();

    } catch (error) {
        console.error('Feedback submission failed:', error);
        showToast('Failed to submit feedback. Please try again.', 'error');
    }
}

// Show toast notification
function showToast(message, type = 'success') {
    const toastId = type === 'success' ? 'successToast' : 'errorToast';
    const toast = document.getElementById(toastId);
    const toastBody = toast.querySelector('.toast-body');

    // Update message
    const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';
    toastBody.innerHTML = `<i class="fas ${icon} me-2"></i>${message}`;

    // Show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

// Utility function to format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Utility function to validate form
function validateForm() {
    const requiredFields = ['amount'];
    for (const fieldId of requiredFields) {
        const field = document.getElementById(fieldId);
        if (!field.value || field.value <= 0) {
            field.focus();
            showToast(`Please enter a valid ${fieldId.replace(/([A-Z])/g, ' $1').toLowerCase()}.`, 'error');
            return false;
        }
    }
    return true;
}

// Add form validation to submit handler
const originalHandleTransactionSubmit = handleTransactionSubmit;
handleTransactionSubmit = async function (event) {
    event.preventDefault();

    if (!validateForm()) {
        return;
    }

    return originalHandleTransactionSubmit.call(this, event);
};

// Enhanced error handling for network issues
window.addEventListener('online', function () {
    showToast('Connection restored. You can now analyze transactions.', 'success');
    checkAPIHealth();
});

window.addEventListener('offline', function () {
    showToast('Connection lost. Please check your internet connection.', 'error');
});

// Add keyboard shortcuts
document.addEventListener('keydown', function (event) {
    // Ctrl+Enter to analyze transaction
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        if (validateForm()) {
            transactionForm.dispatchEvent(new Event('submit'));
        }
    }

    // Ctrl+G to generate sample
    if (event.ctrlKey && event.key === 'g') {
        event.preventDefault();
        generateSampleTransaction();
    }
});

// Add tooltips for better UX
document.addEventListener('DOMContentLoaded', function () {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
