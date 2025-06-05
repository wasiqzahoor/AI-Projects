from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define the path for the CSV file and the model files
CSV_FILE = 'diseases.csv' # Ensure 'diseases.csv' is enclosed in quotes
MODEL_FILE = 'disease_predictor_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

# Global variables for the disease database and the trained ML model
disease_database = {}
disease_predictor_pipeline = None

def load_and_train_model():
    """
    Loads disease data from CSV, trains an ML model, and saves it.
    If model and vectorizer files exist, it loads them instead of retraining.
    """
    global disease_database, disease_predictor_pipeline
    print(f"Attempting to load data from {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: The '{CSV_FILE}' file was not found. Please ensure it is in the same directory as this Python script.")
        # If CSV is mandatory, you might want to exit or raise an exception here
        # For demonstration, we'll proceed with an empty database and no model if CSV is missing
        return
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Successfully loaded {len(df)} rows from {CSV_FILE}.")
        # Populate the disease_database dictionary
        for index, row in df.iterrows():
            disease_name = row['Disease']
            disease_database[disease_name] = {
                "symptoms": row['Symptoms'], # This will be the training text
                "precautions": row['Precautions'],
                "medicines": row['Medicines'].split('; ') if pd.notna(row['Medicines']) else [],
                "diet_plan": row['Diet Plan'].split('; ') if pd.notna(row['Diet Plan']) else [],
                "food_restrictions": row['Food Restrictions'].split('; ') if pd.notna(row['Food Restrictions']) else []
            }
        print("Disease database populated.")
        # Check if trained model and vectorizer exist
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            print("Loading pre-trained model and vectorizer...")
            disease_predictor_pipeline = joblib.load(MODEL_FILE)
            # The vectorizer is part of the pipeline, so no need to load separately.
            print("Model and vectorizer loaded successfully.")
        else:
            print("Training new ML model...")
            # Prepare data for ML model
            X = df['Symptoms'] # Input text (symptom descriptions)
            y = df['Disease']   # Target (disease name)
            # Create a pipeline: TF-IDF Vectorizer + Logistic Regression Classifier
            # Logistic Regression is a good baseline for text classification
            disease_predictor_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
                ('clf', LogisticRegression(max_iter=1000))
            ])
            # Train the model
            disease_predictor_pipeline.fit(X, y)
            print("ML model trained successfully.")
            # Save the trained model and vectorizer
            joblib.dump(disease_predictor_pipeline, MODEL_FILE)
            print("Model and vectorizer saved.")
    except Exception as e:
        print(f"An error occurred during model loading/training: {e}")
        # Ensure disease_database is empty or has a default if training fails
        disease_database = {}
        disease_predictor_pipeline = None

# Load and train the model when the Flask app starts
load_and_train_model()

@app.route('/')
def index():
    """
    Serves the HTML frontend for the AI Health Analyzer.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Health Analyzer</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #a8dadc 0%, #457b9d 100%);
                min-height: 100vh;
            }
            .container {
                background-color: #f1faee;
                border-radius: 1.5rem;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
                max-width: 900px;
                width: 100%;
                padding: 2.5rem;
                display: flex;
                flex-direction: column;
            }
            .section-title {
                color: #1d3557; /* Dark blue */
                font-weight: 600;
                margin-bottom: 1rem;
            }
            .input-field {
                padding: 0.75rem 1rem;
                border: 1px solid #a8dadc;
                border-radius: 0.75rem;
                background-color: #ffffff;
            }
            .input-field:focus {
                outline: none;
                border-color: #457b9d;
                box-shadow: 0 0 0 3px rgba(69, 123, 157, 0.2);
            }
            .btn {
                padding: 0.75rem 1.5rem;
                border-radius: 0.75rem;
                font-weight: 500;
                color: white;
                cursor: pointer;
                transition: all 0.2s ease-in-out;
            }
            .btn-primary {
                background: linear-gradient(45deg, #e63946, #f4a261); /* Red-orange gradient */
            }
            .btn-primary:hover {
                background: linear-gradient(45deg, #f4a261, #e63946);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15), 0 3px 6px -1px rgba(0, 0, 0, 0.08);
            }
            .btn-secondary {
                background-color: #457b9d; /* Dark blue */
            }
            .btn-secondary:hover {
                background-color: #1d3557; /* Even darker blue */
                transform: translateY(-2px);
                box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15), 0 3px 6px -1px rgba(0, 0, 0, 0.08);
            }
            .btn-green {
                background-color: #2a9d8f; /* Teal */
            }
            .btn-green:hover {
                background-color: #264653; /* Darker teal */
                transform: translateY(-2px);
                box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.15), 0 3px 6px -1px rgba(0, 0, 0, 0.08);
            }
            .message-box {
                background-color: #ffe8d6; /* Light peach for default error-like style */
                border: 1px solid #f4a261;
                color: #e63946;
                padding: 1rem;
                border-radius: 0.75rem;
                margin-top: 1rem;
                margin-bottom: 1rem;
                font-weight: 500;
            }
            .loading-spinner {
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-top: 4px solid #fff;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                margin-left: 10px;
                display: inline-block;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .history-entry {
                background-color: #e0f2f7;
                padding: 1rem;
                border-radius: 0.75rem;
                margin-bottom: 0.75rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                border-left: 5px solid #457b9d;
            }
            .history-entry strong {
                color: #1d3557;
            }
            .history-entry p {
                color: #333;
                font-size: 0.95rem;
            }
            .recommendation-box {
                padding: 1rem;
                border-radius: 0.75rem;
                margin-top: 1rem;
                font-weight: 500;
            }
            .recommendation-box ul, .recommendation-box ol {
                list-style-position: inside;
                margin-left: 0;
            }
            /* Modal Styles */
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.6);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease-out, visibility 0.3s ease-out;
            }
            .modal-overlay.show {
                opacity: 1;
                visibility: visible;
            }
            .modal-content {
                background-color: #f1faee;
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
                width: 90%;
                max-width: 500px;
                text-align: center;
                transform: scale(0.95);
                opacity: 0;
                transition: transform 0.3s ease-out, opacity 0.3s ease-out;
            }
            .modal-overlay.show .modal-content {
                transform: scale(1);
                opacity: 1;
            }
            .modal-content h3 {
                color: #1d3557;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            .modal-content textarea, .modal-content input[type="text"] {
                min-height: 40px;
                margin-bottom: 1rem;
                width: 100%;
            }
            /* Section fade-in animation */
            .animated-section {
                opacity: 0;
                transform: translateY(20px);
                transition: opacity 0.6s ease-out, transform 0.6s ease-out;
            }
            .animated-section.show {
                opacity: 1;
                transform: translateY(0);
            }
            #disclaimerOverlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 2000;
                opacity: 1;
                visibility: visible;
                transition: opacity 0.5s ease-out, visibility 0.5s ease-out;
            }
            #disclaimerOverlay.hidden {
                opacity: 0;
                visibility: hidden;
            }
            #disclaimerContent {
                background-color: #f1faee;
                padding: 3rem;
                border-radius: 1.5rem;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
                width: 90%;
                max-width: 600px;
                text-align: center;
            }
            #disclaimerContent h2 {
                color: #e63946;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
            }
            #disclaimerContent p {
                color: #1d3557;
                font-size: 1.1rem;
                line-height: 1.6;
                margin-bottom: 2rem;
            }
            #disclaimerOkBtn {
                background: linear-gradient(45deg, #2a9d8f, #264653);
            }
        </style>
    </head>
    <body class="flex items-center justify-center min-h-screen p-5">
        <div id="disclaimerOverlay">
            <div id="disclaimerContent">
                <h2>Important Disclaimer</h2>
                <p>
                    This system provides preliminary information only and is not a substitute for professional medical advice.
                    The prediction logic and medical recommendations are simplified for demonstration purposes.
                    <strong>Always consult a qualified healthcare professional for diagnosis and treatment.</strong>
                </p>
                <button id="disclaimerOkBtn" class="btn">I Understand & Agree</button>
            </div>
        </div>
        <div class="container" style="display: none;"> <h1 class="text-4xl font-bold text-center text-[#1d3557] mb-6">
                <span class="text-[#e63946]">AI</span> Health Analyzer
            </h1>
            <p class="text-center text-gray-700 mb-6">
                *This system provides preliminary information only and is not a substitute for professional medical advice.
                The prediction logic and medical recommendations are simplified for demonstration.
                **Always consult a qualified healthcare professional for diagnosis and treatment.**
            </p>
            <div id="messageDisplay" class="message-box hidden"></div>
            <div id="inputModeSelection" class="bg-white p-6 rounded-xl shadow-md animated-section">
                <h2 class="section-title text-2xl">Choose Input Mode</h2>
                <p class="text-gray-600 mb-4">Would you like to use voice or text input?</p>
                <div class="flex flex-col sm:flex-row gap-4">
                    <button id="textInputBtn" class="btn btn-secondary flex-1">Text Input</button>
                </div>
            </div>
            <div id="userDetailsSection" class="bg-white p-6 rounded-xl shadow-md hidden animated-section">
                <h2 class="section-title text-2xl">Patient Details</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div>
                        <label for="userName" class="block text-gray-700 text-sm font-medium mb-1">Name:</label>
                        <input type="text" id="userName" class="input-field w-full" placeholder="Enter name" required>
                    </div>
                    <div>
                        <label for="userAge" class="block text-gray-700 text-sm font-medium mb-1">Age:</label>
                        <input type="number" id="userAge" class="input-field w-full" placeholder="Enter age" required min="0" max="150">
                    </div>
                    <div>
                        <label for="userLocation" class="block text-gray-700 text-sm font-medium mb-1">Location:</label>
                        <input type="text" id="userLocation" class="input-field w-full" placeholder="Enter location">
                    </div>
                     <div>
                        <label for="userPhone" class="block text-gray-700 text-sm font-medium mb-1">Phone Number:</label>
                        <input type="text" id="userPhone" class="input-field w-full"
                               pattern="[0-9]{11}"
                               title="Please enter exactly 11 digits"
                               placeholder="e.g., 03001234567" maxlength="11">
                    </div>
                    <div>
                        <label for="userCNIC" class="block text-gray-700 text-sm font-medium mb-1">CNIC Number:</label>
                        <input type="text" id="userCNIC" class="input-field w-full"
                               pattern="[0-9]{13}"
                               title="Please enter exactly 13 digits"
                               placeholder="e.g., 1234567890123" maxlength="13">
                    </div>
                    <div>
                        <label for="userGender" class="block text-gray-700 text-sm font-medium mb-1">Gender:</label>
                        <select id="userGender" class="input-field w-full" required>
                            <option value="">Select Gender</option>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                </div>
                <button id="submitDetailsBtn" class="btn btn-primary w-full">Submit Details</button>
            </div>
            <div id="consultationSection" class="bg-white p-6 rounded-xl shadow-md hidden animated-section">
                <h2 class="section-title text-2xl">Describe Your Problem</h2>
                <textarea id="userDescription" class="input-field w-full h-32 mb-4 resize-y" placeholder="Describe your symptoms here..."></textarea>
                <div class="flex flex-col sm:flex-row gap-4 mb-4">
                    <button id="describeProblemBtn" class="btn btn-primary flex-1">
                        <span id="describeProblemText">Analyze Problem</span>
                        <div id="describeProblemSpinner" class="loading-spinner hidden"></div>
                    </button>
                </div>
                <div id="predictionResult" class="message-box hidden text-center text-xl font-semibold" style="background-color: #e0f2f7; border-color: #457b9d; color: #1d3557;">
                    <span class="text-[#e63946]">Predicted Condition:</span> <span id="predictedConditionText"></span>
                </div>
                <div id="precautionsDisplay" class="recommendation-box hidden bg-blue-100 border-blue-500 text-blue-700 mt-4">
                    <strong class="text-[#1d3557]">Precautions:</strong> <span id="precautionsText"></span>
                </div>
                <div id="medicineRecommendations" class="recommendation-box hidden bg-green-100 border-green-500 text-green-700 mt-4">
                    <strong class="text-[#1d3557]">Medicine Recommendations (Example Course):</strong>
                    <ul id="medicineList" class="mt-2 list-disc"></ul>
                </div>
                <div id="dietPlanDisplay" class="recommendation-box hidden bg-yellow-100 border-yellow-500 text-yellow-700 mt-4">
                    <strong class="text-[#1d3557]">Diet Plan (General Guidelines):</strong>
                    <ul id="dietList" class="mt-2 list-disc"></ul>
                </div>
                <div id="foodRestrictionsDisplay" class="recommendation-box hidden bg-red-100 border-red-500 text-red-700 mt-4">
                    <strong class="text-[#1d3557]">Food Restrictions:</strong>
                    <ul id="restrictionsList" class="mt-2 list-disc"></ul>
                </div>
                <div id="feedbackSection" class="mt-6 p-4 bg-[#fef9e7] rounded-xl shadow-inner hidden">
                    <p class="text-gray-700 text-center mb-4">Was this information helpful?</p>
                    <div class="flex justify-center gap-4">
                        <button id="feedbackYesBtn" class="btn btn-green px-6 py-2">Yes</button>
                        <button id="feedbackNoBtn" class="btn btn-primary px-6 py-2">No</button>
                    </div>
                </div>
                <button id="anotherConsultationBtn" class="btn btn-secondary w-full mt-6 hidden">
                    Describe Another Problem for This Patient
                </button>
                <button id="newPatientBtn" class="btn btn-primary w-full mt-4 hidden">
                    Start New Patient Consultation
                </button>
            </div>
            <div id="patientHistorySection" class="bg-white p-6 rounded-xl shadow-md hidden animated-section">
                <h2 class="section-title text-2xl">Patient History</h2>
                <div id="historyList" class="max-h-96 overflow-y-auto pr-2">
                    <p class="text-gray-600 text-center" id="noHistoryMessage">No consultations recorded yet.</p>
                </div>
                <button id="downloadHistoryBtn" class="btn btn-green w-full mt-6">
                    Download Full History
                </button>
            </div>
            <div id="followUpQuestionModal" class="modal-overlay hidden">
                <div class="modal-content">
                    <h3 id="modalQuestionText"></h3>
                    <input type="text" id="modalAnswerInput" class="input-field w-full mb-4" placeholder="Type your answer here...">
                    <div class="flex flex-col sm:flex-row gap-4 justify-center">
                        <button id="modalSubmitBtn" class="btn btn-primary flex-1">
                            Submit
                            <div id="modalSpinner" class="loading-spinner hidden"></div>
                        </button>
                    </div>
                </div>
            </div>
            <footer class="text-center text-gray-600 text-sm mt-8">
                © 2025 AI Health Analyzer. All rights reserved.
            </footer>
        </div>
        <script>
        // --- Global Variables and Constants ---
        const PATIENT_RECORDS_KEY = 'healthAnalyzerPatientRecords';
        let currentPatient = null;
        let inputMode = 'text'; // Default to text, voice input removed
        let synth = window.speechSynthesis;

        // --- DOM Elements ---
        const messageDisplay = document.getElementById('messageDisplay');
        const inputModeSelection = document.getElementById('inputModeSelection');
        const textInputBtn = document.getElementById('textInputBtn');
        const userDetailsSection = document.getElementById('userDetailsSection');
        const userNameInput = document.getElementById('userName');
        const userAgeInput = document.getElementById('userAge');
        const userLocationInput = document.getElementById('userLocation');
        const userPhoneInput = document.getElementById('userPhone');
        const userCNICInput = document.getElementById('userCNIC');
        const userGenderSelect = document.getElementById('userGender');
        const submitDetailsBtn = document.getElementById('submitDetailsBtn');
        const consultationSection = document.getElementById('consultationSection');
        const userDescriptionTextarea = document.getElementById('userDescription');
        const describeProblemBtn = document.getElementById('describeProblemBtn');
        const describeProblemText = document.getElementById('describeProblemText');
        const describeProblemSpinner = document.getElementById('describeProblemSpinner');
        const predictionResult = document.getElementById('predictionResult');
        const predictedConditionText = document.getElementById('predictedConditionText');
        const precautionsDisplay = document.getElementById('precautionsDisplay');
        const precautionsText = document.getElementById('precautionsText');
        const medicineRecommendations = document.getElementById('medicineRecommendations');
        const medicineList = document.getElementById('medicineList');
        const dietPlanDisplay = document.getElementById('dietPlanDisplay');
        const dietList = document.getElementById('dietList');
        const foodRestrictionsDisplay = document.getElementById('foodRestrictionsDisplay');
        const restrictionsList = document.getElementById('restrictionsList');
        const feedbackSection = document.getElementById('feedbackSection');
        const feedbackYesBtn = document.getElementById('feedbackYesBtn');
        const feedbackNoBtn = document.getElementById('feedbackNoBtn');
        const anotherConsultationBtn = document.getElementById('anotherConsultationBtn');
        const newPatientBtn = document.getElementById('newPatientBtn');
        const patientHistorySection = document.getElementById('patientHistorySection');
        const historyList = document.getElementById('historyList');
        const noHistoryMessage = document.getElementById('noHistoryMessage');
        const downloadHistoryBtn = document.getElementById('downloadHistoryBtn');
        const followUpQuestionModal = document.getElementById('followUpQuestionModal');
        const modalQuestionText = document.getElementById('modalQuestionText');
        const modalAnswerInput = document.getElementById('modalAnswerInput');
        const modalSubmitBtn = document.getElementById('modalSubmitBtn');
        const modalSpinner = document.getElementById('modalSpinner');
        const disclaimerOverlay = document.getElementById('disclaimerOverlay');
        const disclaimerOkBtn = document.getElementById('disclaimerOkBtn');

        // --- Helper Functions ---
        function showMessage(message, isError = false) {
            messageDisplay.textContent = message;
            messageDisplay.classList.remove('hidden', 'bg-ffe8d6', 'border-f4a261', 'text-e63946', 'bg-green-100', 'border-green-500', 'text-green-700', 'bg-blue-100', 'border-blue-500', 'text-blue-700', 'bg-yellow-100', 'border-yellow-500', 'text-yellow-700', 'bg-red-100', 'border-red-500', 'text-red-700');
            if (isError) {
                messageDisplay.classList.add('bg-ffe8d6', 'border-f4a261', 'text-e63946');
                // Default message-box error style
            } else {
                // Success message uses a light green style, similar to recommendation boxes but for general messages
                messageDisplay.style.backgroundColor = '#e6f7e9';
                messageDisplay.style.color = '#264653';
                messageDisplay.style.borderColor = '#2a9d8f';
            }
            messageDisplay.classList.remove('hidden');
            setTimeout(() => {
                messageDisplay.classList.add('hidden');
            }, 5000);
        }

        function speak(text) {
            if (!synth || !window.speechSynthesis) {
                console.warn("SpeechSynthesis not supported in this browser.");
                return;
            }
            synth.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            utterance.rate = 0.9;
            utterance.volume = 0.9;
            synth.speak(utterance);
            return utterance; // Return utterance to attach onend listeners if needed
        }

        // Removed getVoiceInput function entirely as voice input is disabled.

        function getUserInput(promptText, callback) {
            modalQuestionText.textContent = promptText;
            modalAnswerInput.value = '';
            followUpQuestionModal.classList.add('show');
            // Ensure only text input is used for the modal
            modalSubmitBtn.childNodes[0].nodeValue = 'Submit ';
            speak(promptText); // Speak the modal's question
            resolveModalPromise = (response) => {
                followUpQuestionModal.classList.remove('show');
                callback(response);
            };
        }

        function loadPatientRecords() {
            try {
                const records = JSON.parse(localStorage.getItem(PATIENT_RECORDS_KEY) || '{}');
                return records;
            } catch (e) {
                console.error("Error loading patient records:", e);
                return {};
            }
        }

        function savePatientRecords(records) {
            try {
                localStorage.setItem(PATIENT_RECORDS_KEY, JSON.stringify(records));
            } catch (e) {
                console.error("Error saving patient records:", e);
                showMessage("Error saving data. Browser storage might be full or disabled.", true);
            }
        }

        function updatePatientHistoryDisplay() {
            const records = loadPatientRecords();
            const patientKey = currentPatient && typeof currentPatient === 'object' && currentPatient.name ? currentPatient.name : null;
            const patientData = patientKey ? records[patientKey] : null;
            historyList.innerHTML = '';
            if (patientData && patientData.consultations && patientData.consultations.length > 0) {
                noHistoryMessage.classList.add('hidden');
                patientData.consultations.forEach((consultation, index) => {
                    const entryDiv = document.createElement('div');
                    entryDiv.className = 'history-entry';
                    let medicinesHtml = consultation.medicines && consultation.medicines.length > 0 ? consultation.medicines.map(m => `<li>${m}</li>`).join('') : '<li>N/A</li>';
                    let dietHtml = consultation.diet_plan && consultation.diet_plan.length > 0 ? consultation.diet_plan.map(d => `<li>${d}</li>`).join('') : '<li>N/A</li>';
                    let restrictionsHtml = consultation.food_restrictions && consultation.food_restrictions.length > 0 ? consultation.food_restrictions.map(r => `<li>${r}</li>`).join('') : '<li>N/A</li>';
                    entryDiv.innerHTML = `
                        <p class="font-bold text-lg text-blue-800">Patient: ${patientData.details.name || 'N/A'}</p>
                        <ul class="list-disc list-inside ml-4 mb-2 text-sm">
                            <li>Age: ${patientData.details.age || 'N/A'}</li>
                            <li>Location: ${patientData.details.location || 'N/A'}</li>
                            <li>Phone: ${patientData.details.phone || 'N/A'}</li>
                            <li>CNIC: ${patientData.details.cnic || 'N/A'}</li>
                            <li>Gender: ${patientData.details.gender || 'N/A'}</li>
                        </ul>
                        <p><strong>Consultation ${index + 1}</strong> (${consultation.timestamp || 'N/A'})</p>
                        <p><strong>Description:</strong> ${consultation.description || 'N/A'}</p>
                        <p><strong>Predicted:</strong> ${consultation.predicted_condition || 'N/A'}</p>
                        <p><strong>Precautions:</strong> ${consultation.precautions || 'N/A'}</p>
                        <p><strong>Feedback:</strong> ${consultation.feedback || 'Pending'}</p>
                        <p><strong>Extracted Symptoms (for history):</strong></p>
                        <ul class="list-disc list-inside ml-4 text-sm">
                            ${Object.entries(consultation.extracted_symptoms || {})
                                .filter(([key, value]) => value !== 0 && value !== 'unknown' && value !== false && value !== null && value !== undefined)
                                .map(([key, value]) => `<li>${key.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase())}: ${value}</li>`)
                                .join('') || '<li>No specific symptoms extracted.</li>'
                            }
                        </ul>
                        <p class="mt-2"><strong>Medicines:</strong></p><ul class="list-disc list-inside ml-4 text-sm">${medicinesHtml}</ul>
                        <p class="mt-2"><strong>Diet Plan:</strong></p><ul class="list-disc list-inside ml-4 text-sm">${dietHtml}</ul>
                        <p class="mt-2"><strong>Food Restrictions:</strong></p><ul class="list-disc list-inside ml-4 text-sm">${restrictionsHtml}</ul>
                    `;
                    historyList.appendChild(entryDiv);
                });
            } else {
                noHistoryMessage.classList.remove('hidden');
                if (patientKey) {
                    noHistoryMessage.textContent = `No consultations recorded yet for ${patientKey}.`;
                } else {
                    noHistoryMessage.textContent = "No patient selected or no history available.";
                }
            }
        }

        // MOCK_FEATURE_COLS and extractStructuredSymptoms are kept for historical logging,
        // but the actual prediction input to the ML model will be the raw userDescription.
        // For faster interaction, we are simplifying or removing follow-up questions.
        const MOCK_FEATURE_COLS = [
            "fever", "cough", "fatigue", "headache", "nausea",
            "shortness_of_breath", "chest_pain", "sore_throat", "body_aches",
            "diarrhea_present", "fever_present", "cough_type", "fever_duration",
            "fever_severity", "headache_type", "chest_pain_type", "nausea_vomiting",
            "nausea_duration", "fatigue_impact", "allergic_rhinitis_specific",
            "uti_specific", "anxiety_symptoms_present", "gerd_symptoms_present",
            "gastritis_symptoms_present"
        ];

        function extractStructuredSymptoms(userDescription) {
            const symptomsData = {};
            MOCK_FEATURE_COLS.forEach(col => symptomsData[col] = 0);
            symptomsData['cough_type'] = 'unknown';
            symptomsData['fever_present'] = 0;
            symptomsData['fever_duration'] = 'unknown';
            symptomsData['fever_severity'] = 'unknown';
            symptomsData['headache_type'] = 'unknown';
            symptomsData['chest_pain_type'] = 'unknown';
            symptomsData['nausea_vomiting'] = 0;
            symptomsData['nausea_duration'] = 'unknown';
            symptomsData['fatigue_impact'] = 'unknown';
            symptomsData['sore_throat'] = 0;
            symptomsData['body_aches'] = 0;
            symptomsData['diarrhea_present'] = 0;
            const descLower = userDescription.toLowerCase();
            if (/(fever|temperature|high temp|elevated temp|hot|feverish|burning up|chills)/.test(descLower)) {
                symptomsData["fever"] = 1;
                symptomsData['fever_present'] = 1;
            }
            if (/(cough|coughing|cold|hacking|wheezing|throat tickle|bronchitis|respiratory)/.test(descLower)) {
                symptomsData["cough"] = 1;
                if (/(dry cough|dry hacking|not producing mucus)/.test(descLower)) symptomsData['cough_type'] = 'dry';
                else if (/(wet cough|productive cough|mucus|phlegm|coughing stuff up)/.test(descLower)) symptomsData['cough_type'] = 'wet';
            }
            if (/(fatigue|tired|exhausted|weary|drained|low energy|lethargic|sleepy)/.test(descLower)) {
                symptomsData["fatigue"] = 1;
            }
            if (/(headache|migraine|head pain|skull ache|temple pain|forehead pain|dizziness)/.test(descLower)) {
                symptomsData["headache"] = 1;
                if (/(throbbing headache|pulsating headache|migraine)/.test(descLower)) symptomsData['headache_type'] = 'throbbing/migraine';
                else if (/(dull ache|aching)/.test(descLower)) symptomsData['headache_type'] = 'dull/aching';
                else if (/(sharp|stabbing)/.test(descLower)) symptomsData['headache_type'] = 'sharp';
                else if (/(pressure|tightness)/.test(descLower)) symptomsData['headache_type'] = 'pressure';
                else symptomsData['headache_type'] = 'unspecified';
            }
            if (/(nausea|sick to my stomach|queasy)/.test(descLower)) {
                symptomsData["nausea"] = 1;
            }
            if (/(vomit|throwing up|puking)/.test(descLower)) {
                symptomsData["nausea_vomiting"] = 1;
            }
            if (/(diarrhea|loose stools|runs)/.test(descLower)) {
                symptomsData["diarrhea_present"] = 1;
            }
            if (/(short of breath|breathing difficulty|breathless|gasping for air|short winded|can't catch my breath|wheezing)/.test(descLower)) {
                symptomsData["shortness_of_breath"] = 1;
            }
            if (/(chest pain|chest discomfort|chest tightness|pressure in chest|squeezing chest|heart pain)/.test(descLower)) {
                symptomsData["chest_pain"] = 1;
                if (/(sharp chest pain|stabbing chest pain)/.test(descLower)) symptomsData['chest_pain_type'] = 'sharp/stabbing';
                else if (/(dull chest pain|aching chest)/.test(descLower)) symptomsData['chest_pain_type'] = 'dull/aching';
                else if (/(pressure in chest)/.test(descLower)) symptomsData['chest_pain_type'] = 'pressure';
                else symptomsData['chest_pain_type'] = 'localized';
            }
            if (/(sore throat|throat pain|scratchy throat)/.test(descLower)) {
                symptomsData["sore_throat"] = 1;
            }
            if (/(body aches|muscle pain|joint pain)/.test(descLower)) {
                symptomsData["body_aches"] = 1;
            }
            if (/(itchy nose|itchy eyes|watery eyes)/.test(descLower)) {
                symptomsData["allergic_rhinitis_specific"] = 1;
            }
            if (/(burning urination|painful urination|frequent urination)/.test(descLower)) {
                 symptomsData["uti_specific"] = 1;
            }
            if (/(worry|anxious|restless|on edge|difficulty concentrating|irritable|muscle tension|sleep problems)/.test(descLower)) {
                symptomsData["anxiety_symptoms_present"] = 1;
            }
            if (/(heartburn|acid reflux|regurgitation|sour taste)/.test(descLower)) {
                symptomsData["gerd_symptoms_present"] = 1;
            }
            if (/(stomach pain|burning stomach|nausea|vomiting|bloating|indigestion)/.test(descLower)) {
                symptomsData["gastritis_symptoms_present"] = 1;
            }
            return symptomsData;
        }

        // Optimized: Simplified follow-up questions for faster interaction
        async function askFollowUpQuestion(symptomsData) {
            let currentSymptoms = { ...symptomsData };
            // For faster interaction, we'll only ask one general clarifying question if needed.
            // Or, you can remove this function entirely if you want instant results without any follow-up.
            // For now, let's keep one generic clarifying question.
            if (Object.values(currentSymptoms).every(val => val === 0 || val === 'unknown')) {
                // If no specific symptoms were extracted, ask a general clarifying question
                let response = await new Promise(resolve => {
                    getUserInput("Could you provide more details about the onset or severity of your main symptom?", (res) => {
                        resolve(res);
                    });
                });
                if (response && response.trim() !== "") {
                    // This response can be used to update a 'clarifying_details' field for history
                    currentSymptoms['clarifying_details'] = response;
                } else {
                    console.warn("No response or empty response for clarifying question.");
                }
            }
            return currentSymptoms;
        }

        // --- Event Listeners for real-time speech output ---
        userNameInput.addEventListener('input', () => speak(userNameInput.value));
        userAgeInput.addEventListener('input', () => speak(userAgeInput.value));
        userLocationInput.addEventListener('input', () => speak(userLocationInput.value));
        userPhoneInput.addEventListener('input', () => speak(userPhoneInput.value));
        userCNICInput.addEventListener('input', () => speak(userCNICInput.value));
        userGenderSelect.addEventListener('change', () => speak(userGenderSelect.value));
        userDescriptionTextarea.addEventListener('input', () => speak(userDescriptionTextarea.value));


        // --- Event Handlers ---
        disclaimerOkBtn.addEventListener('click', () => {
            synth.cancel();
            disclaimerOverlay.classList.add('hidden');
            document.querySelector('.container').style.display = 'flex'; // Show main content
            inputModeSelection.classList.add('show');
            // Directly transition to text input, removing the choice
            inputModeSelection.classList.remove('show');
            inputModeSelection.classList.add('hidden');
            userDetailsSection.classList.remove('hidden');
            userDetailsSection.classList.add('show');
            speak("Please enter your details to begin.");
        });

        // Removed voiceInputBtn listener as voice input is disabled.

        textInputBtn.addEventListener('click', () => {
            inputMode = 'text';
            inputModeSelection.classList.remove('show');
            inputModeSelection.classList.add('hidden');
            userDetailsSection.classList.remove('hidden');
            userDetailsSection.classList.add('show');
            speak("Please enter your details to begin.");
        });

        submitDetailsBtn.addEventListener('click', async () => {
            const name = userNameInput.value.trim();
            const age = parseInt(userAgeInput.value);
            const location = userLocationInput.value.trim();
            const phone = userPhoneInput.value.trim();
            const cnic = userCNICInput.value.trim();
            const gender = userGenderSelect.value;

            // Basic validation
            if (!name || !age || isNaN(age) || age <= 0 || !gender) {
                showMessage("Please fill in all patient details correctly, including a valid age and gender.", true);
                speak("Please fill in all patient details correctly, including a valid age and gender.");
                return;
            }

            // Phone number validation: must be 11 digits and numeric
            if (!/^[0-9]{11}$/.test(phone)) {
                showMessage("Please enter a valid phone number, exactly 11 digits long.", true);
                speak("Please enter a valid phone number, exactly 11 digits long.");
                return;
            }

            // CNIC validation: must be 13 digits and numeric
            if (!/^[0-9]{13}$/.test(cnic)) {
                showMessage("Please enter a valid CNIC number, exactly 13 digits long.", true);
                speak("Please enter a valid CNIC number, exactly 13 digits long.");
                return;
            }

            // Location validation: must be a string and not empty
            if (typeof location !== 'string' || location.length === 0) {
                showMessage("Please enter a valid location.", true);
                speak("Please enter a valid location.");
                return;
            }

            let records = loadPatientRecords();
            currentPatient = { name, age, location, phone, cnic, gender };
            if (!records[name]) {
                records[name] = { details: { ...currentPatient }, consultations: [] };
                showMessage(`New patient record created for ${name}.`);
                speak(`New patient record created for ${name}.`);
            } else {
                records[name].details = { ...currentPatient };
                showMessage(`Welcome back, ${name}! Your details have been updated.`);
                speak(`Welcome back, ${name}!`);
            }
            savePatientRecords(records);
            userDetailsSection.classList.remove('show');
            userDetailsSection.classList.add('hidden');
            consultationSection.classList.remove('hidden');
            consultationSection.classList.add('show');
            patientHistorySection.classList.remove('hidden');
            patientHistorySection.classList.add('show');
            updatePatientHistoryDisplay();
            speak("Now, please describe your health problem or symptoms.");
        });

        describeProblemBtn.addEventListener('click', async () => {
            const description = userDescriptionTextarea.value.trim();
            if (!description) {
                showMessage("Please describe your symptoms first.", true);
                speak("Please describe your symptoms first.");
                return;
            }
            userDescriptionTextarea.focus();
            await processDescription(description);
        });

        // Removed voiceDescribeProblemBtn listener as voice input is disabled.

        async function processDescription(description) {
            describeProblemText.classList.add('hidden');
            describeProblemSpinner.classList.remove('hidden');
            describeProblemBtn.disabled = true;

            try {
                // The ML model in the backend will directly take the raw description.
                // We still use extractStructuredSymptoms here for the purpose of saving
                // a detailed symptom breakdown in the patient history, which is useful for review.
                // Simplified follow-up questions for faster interaction
                let initialSymptomsForHistory = extractStructuredSymptoms(description);
                let updatedSymptomsForHistory = await askFollowUpQuestion(initialSymptomsForHistory); // This will be faster now
                // Send raw description to Flask backend for ML prediction
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ description: description }), // Send raw description
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`HTTP error! status: ${response.status} - ${errorData.error || response.statusText}`);
                }
                const data = await response.json();
                const predictedCondition = data.predicted_condition;
                const precautions = data.precautions;
                const medicalAdvice = {
                    medicines: data.medicines || [],
                    diet_plan: data.diet_plan || [],
                    food_restrictions: data.food_restrictions || []
                };

                predictedConditionText.textContent = predictedCondition;
                precautionsText.textContent = precautions;
                predictionResult.classList.remove('hidden');
                precautionsDisplay.classList.remove('hidden');

                medicineList.innerHTML = '';
                if (medicalAdvice.medicines && medicalAdvice.medicines.length > 0) {
                    medicalAdvice.medicines.forEach(item => medicineList.appendChild(Object.assign(document.createElement('li'), { textContent: item })));
                    medicineRecommendations.classList.remove('hidden');
                } else { medicineRecommendations.classList.add('hidden'); }

                dietList.innerHTML = '';
                if (medicalAdvice.diet_plan && medicalAdvice.diet_plan.length > 0) {
                    medicalAdvice.diet_plan.forEach(item => dietList.appendChild(Object.assign(document.createElement('li'), { textContent: item })));
                    dietPlanDisplay.classList.remove('hidden');
                } else { dietPlanDisplay.classList.add('hidden'); }

                restrictionsList.innerHTML = '';
                if (medicalAdvice.food_restrictions && medicalAdvice.food_restrictions.length > 0) {
                    medicalAdvice.food_restrictions.forEach(item => restrictionsList.appendChild(Object.assign(document.createElement('li'), { textContent: item })));
                    foodRestrictionsDisplay.classList.remove('hidden');
                } else { foodRestrictionsDisplay.classList.add('hidden'); }

                let adviceSpeech = `Based on your description, the system suggests: ${predictedCondition}. As a precaution: ${precautions}.`;
                if (medicalAdvice.medicines && medicalAdvice.medicines.length > 0 && medicalAdvice.medicines[0] !== "No specific medication examples. Consult a doctor.") {
                    adviceSpeech += ` For example, a medication like ${medicalAdvice.medicines[0].split(':')[0]} might be considered.`;
                }
                adviceSpeech += " Remember, this is a simplified analysis and not a substitute for professional medical advice.";
                speak(adviceSpeech);

                if (currentPatient && currentPatient.name) {
                    let records = loadPatientRecords();
                    if (records[currentPatient.name]) {
                        const consultationEntry = {
                            timestamp: new Date().toLocaleString(),
                            description: description,
                            extracted_symptoms: updatedSymptomsForHistory, // Save structured symptoms for history
                            predicted_condition: predictedCondition,
                            precautions: precautions,
                            medicines: medicalAdvice.medicines || [],
                            diet_plan: medicalAdvice.diet_plan || [],
                            food_restrictions: medicalAdvice.food_restrictions || [],
                            feedback: 'Pending'
                        };
                        predictionResult.dataset.consultationIndex = records[currentPatient.name].consultations.length;
                        records[currentPatient.name].consultations.push(consultationEntry);
                        savePatientRecords(records);
                        updatePatientHistoryDisplay();
                    }
                } else {
                    showMessage("No current patient active. Consultation not saved to history.", true);
                }
                feedbackSection.classList.remove('hidden');
            } catch (error) {
                console.error("Error during analysis:", error);
                showMessage(`An error occurred during analysis: ${error.message}. Please try again. Ensure the Flask backend is running and the diseases.csv file is correctly formatted.`, true);
                speak(`An error occurred during analysis. Please try again. Ensure the Flask backend is running and the diseases.csv file is correctly formatted.`);
            } finally {
                describeProblemText.classList.remove('hidden');
                describeProblemSpinner.classList.add('hidden');
                describeProblemBtn.disabled = false;
            }
        }

        function updateFeedback(feedbackValue) {
            if (currentPatient && currentPatient.name && predictionResult.dataset.consultationIndex !== undefined) {
                let records = loadPatientRecords();
                const patientRecord = records[currentPatient.name];
                const consultIndex = parseInt(predictionResult.dataset.consultationIndex);
                if (patientRecord && patientRecord.consultations && patientRecord.consultations[consultIndex]) {
                    patientRecord.consultations[consultIndex].feedback = feedbackValue;
                    savePatientRecords(records);
                    updatePatientHistoryDisplay();
                    showMessage(`Feedback (${feedbackValue}) recorded. Thank you!`, false);
                    speak(`Feedback recorded. Thank you!`);
                } else {
                    showMessage("Could not record feedback: consultation data issue.", true);
                }
            } else {
                showMessage("Could not record feedback: no active consultation index.", true);
            }
            delete predictionResult.dataset.consultationIndex;
            feedbackSection.classList.add('hidden');
            anotherConsultationBtn.classList.remove('hidden');
            newPatientBtn.classList.remove('hidden');
        }

        feedbackYesBtn.addEventListener('click', () => updateFeedback("Helpful"));
        feedbackNoBtn.addEventListener('click', () => updateFeedback("Not Helpful"));

        modalSubmitBtn.addEventListener('click', () => {
            if (resolveModalPromise) {
                const answer = modalAnswerInput.value.trim();
                synth.cancel();
                resolveModalPromise(answer);
                resolveModalPromise = null;
            }
        });

        // Removed modalVoiceInputBtn listener as voice input is disabled.

        anotherConsultationBtn.addEventListener('click', () => {
            userDescriptionTextarea.value = '';
            predictionResult.classList.add('hidden');
            precautionsDisplay.classList.add('hidden');
            medicineRecommendations.classList.add('hidden');
            dietPlanDisplay.classList.add('hidden');
            foodRestrictionsDisplay.classList.add('hidden');
            feedbackSection.classList.add('hidden');
            anotherConsultationBtn.classList.add('hidden');
            newPatientBtn.classList.add('hidden');
            speak("Please describe the new problem or symptoms.");
        });

        newPatientBtn.addEventListener('click', () => {
            currentPatient = null;
            consultationSection.classList.add('hidden');
            consultationSection.classList.remove('show');
            userDetailsSection.classList.add('hidden');
            userDetailsSection.classList.remove('show');
            patientHistorySection.classList.add('hidden');
            patientHistorySection.classList.remove('show');
            userNameInput.value = '';
            userAgeInput.value = '';
            userLocationInput.value = '';
            userPhoneInput.value = '';
            userCNICInput.value = '';
            userGenderSelect.value = '';
            userDescriptionTextarea.value = '';
            predictionResult.classList.add('hidden');
            precautionsDisplay.classList.add('hidden');
            medicineRecommendations.classList.add('hidden');
            dietPlanDisplay.classList.add('hidden');
            foodRestrictionsDisplay.classList.add('hidden');
            feedbackSection.classList.add('hidden');
            anotherConsultationBtn.classList.add('hidden');
            newPatientBtn.classList.add('hidden');
            historyList.innerHTML = '';
            noHistoryMessage.classList.remove('hidden');
            noHistoryMessage.textContent = "No patient selected or no history available.";
            inputModeSelection.classList.remove('hidden');
            inputModeSelection.classList.add('show');
            speak("Starting a new patient consultation. Please choose your input mode.");
        });

        downloadHistoryBtn.addEventListener('click', () => {
            const records = loadPatientRecords();
            let historyContent = `--- Comprehensive Patient Consultation History ---\\n`;
            historyContent += `Generated on: ${new Date().toLocaleString()}\\n\\n`;
            if (Object.keys(records).length === 0) {
                historyContent += "No patient records found.\\n";
            } else {
                for (const patientName in records) {
                    if (records.hasOwnProperty(patientName)) {
                        const data = records[patientName];
                        historyContent += `=================================================\\n`;
                        historyContent += `Patient Name: ${data.details.name || 'N/A'}\\n`;
                        historyContent += `Age: ${data.details.age || 'N/A'}\\n`;
                        historyContent += `Location: ${data.details.location || 'N/A'}\\n`;
                        historyContent += `Phone: ${data.details.phone || 'N/A'}\\n`;
                        historyContent += `CNIC: ${data.details.cnic || 'N/A'}\\n`;
                        historyContent += `Gender: ${data.details.gender || 'N/A'}\\n`;
                        historyContent += `-------------------------------------------------\\n`;
                        historyContent += `Consultations:\\n`;
                        if (data.consultations && data.consultations.length > 0) {
                            data.consultations.forEach((consultation, i) => {
                                historyContent += `  Consultation ${i + 1} (Timestamp: ${consultation.timestamp || 'N/A'}):\\n`;
                                historyContent += `    User Description: ${consultation.description || 'N/A'}\\n`;
                                historyContent += `    Predicted Condition: ${consultation.predicted_condition || 'N/A'}\\n`;
                                historyContent += `    Precautions: ${consultation.precautions || 'N/A'}\\n`;
                                historyContent += `    Feedback: ${consultation.feedback || 'N/A'}\\n`;
                                historyContent += `    Extracted Symptoms (for history):\\n`;
                                const extractedSymptoms = consultation.extracted_symptoms || {};
                                let symptomsLogged = false;
                                for (const symptom in extractedSymptoms) {
                                    if (extractedSymptoms.hasOwnProperty(symptom)) {
                                        const value = extractedSymptoms[symptom];
                                        if (value !== 0 && value !== 'unknown' && value !== false && value !== null && value !== undefined && String(value).trim() !== "") {
                                            historyContent += `      - ${symptom.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase())}: ${value}\\n`;
                                            symptomsLogged = true;
                                        }
                                    }
                                }
                                if (!symptomsLogged) historyContent += `      - No specific symptoms extracted or detailed.\\n`;
                                historyContent += `    Medicine Recommendations:\\n`;
                                if (consultation.medicines && consultation.medicines.length > 0) {
                                    consultation.medicines.forEach(m => historyContent += `      - ${m}\\n`);
                                } else { historyContent += `      N/A\\n`; }
                                historyContent += `    Diet Plan:\\n`;
                                if (consultation.diet_plan && consultation.diet_plan.length > 0) {
                                    consultation.diet_plan.forEach(d => historyContent += `      - ${d}\\n`);
                                } else { historyContent += `      N/A\\n`; }
                                historyContent += `    Food Restrictions:\\n`;
                                if (consultation.food_restrictions && consultation.food_restrictions.length > 0) {
                                    consultation.food_restrictions.forEach(r => historyContent += `      - ${r}\\n`);
                                } else { historyContent += `      N/A\\n`; }
                                historyContent += "\\n";
                            });
                        } else {
                            historyContent += "  No consultations recorded for this patient.\\n";
                        }
                        historyContent += "\\n";
                    }
                }
            }
            historyContent += `--- End of Document ---\\n`;
            const blob = new Blob([historyContent], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `AI_HealthAnalyzer_History_${new Date().toISOString().slice(0, 10)}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showMessage("Patient history download initiated!", false);
            speak("Patient history download initiated!");
        });

        window.onload = () => {
            document.querySelector('.container').style.display = 'none'; // Keep container hidden
            disclaimerOverlay.classList.remove('hidden'); // Ensure disclaimer is visible
            speak("Welcome to the AI Health Analyzer. Important Disclaimer: This system provides preliminary information only and is not a substitute for professional medical advice. The prediction logic and medical recommendations are simplified for demonstration purposes. Always consult a qualified healthcare professional for diagnosis and treatment. Please click 'I Understand and Agree' to proceed.");
        };
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive user's symptom description,
    use the ML model to predict the disease, and return recommendations.
    """
    data = request.get_json()
    user_description = data.get('description', '')
    if not user_description:
        return jsonify({"error": "No symptom description provided"}), 400
    if disease_predictor_pipeline is None:
        # Check if the CSV file exists but training failed
        if os.path.exists(CSV_FILE):
             return jsonify({"error": "ML model failed to load/train. Check backend logs for errors during CSV reading or model training."}), 500
        else:
            return jsonify({"error": f"ML model not available because '{CSV_FILE}' was not found. Please place the CSV file in the same directory as the script."}), 500
    try:
        # Predict the disease using the trained ML model
        predicted_condition = disease_predictor_pipeline.predict([user_description])[0]
        print(f"Predicted condition for '{user_description}': {predicted_condition}")
        # Retrieve details from the loaded disease_database
        disease_info = disease_database.get(predicted_condition, {})
        precautions = disease_info.get("precautions", "Please consult a healthcare professional for specific advice.")
        medicines = disease_info.get("medicines", ["No specific medication examples. Consult a doctor."])
        diet_plan = disease_info.get("diet_plan", ["Maintain a balanced diet. Consult a doctor for specifics."])
        food_restrictions = disease_info.get("food_restrictions", ["Avoid known triggers. Consult a doctor."])
        response = {
            "predicted_condition": predicted_condition,
            "precautions": precautions,
            "medicines": medicines,
            "diet_plan": diet_plan,
            "food_restrictions": food_restrictions
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)
