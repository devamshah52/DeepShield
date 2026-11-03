# DeepShield – AI-Powered Audio Deepfake Detection System

DeepShield is a deep learning-based audio detection system designed to classify audio recordings into three categories: **Genuine, Text-to-Speech (TTS), and Voice Conversion**.

The project uses a trained PyTorch deep learning model to analyze audio features and identify potentially AI-generated or manipulated speech.

---

## 1. Project Overview

With the growth of artificial intelligence, generating realistic synthetic voices and manipulating existing voices has become increasingly accessible.

DeepShield aims to help identify these types of audio by analyzing speech patterns and classifying recordings using a trained deep learning model.

### Detection Categories

| Category | Description |
|---|---|
| Genuine | Real human speech recordings |
| TTS | Speech generated from written text using Text-to-Speech technology |
| Voice Conversion | Existing speech transformed to sound like another speaker |

## 2. Key Features

- User registration and authentication
- Audio file upload
- AI-powered audio classification
- Three-class detection: Genuine, TTS and Voice Conversion
- Prediction confidence and class probabilities
- User dashboard
- Detection history
- Analytics dashboard
- Admin functionality

## 3. Technology Stack

### Frontend
- React.js
- Vite
- React Router
- Axios
- Recharts

### Backend
- Python
- FastAPI
- Uvicorn
- PyMongo
- Pydantic

### Machine Learning
- PyTorch
- Librosa
- NumPy
- DeepShield V3.2 Weighted CNN

### Database
- MongoDB

## 4. System Architecture

DeepShield follows a frontend-backend-machine-learning architecture.

1. React.js provides the user interface.
2. FastAPI handles API requests and application logic.
3. The prediction service preprocesses uploaded audio.
4. Librosa extracts Mel spectrogram features.
5. The trained PyTorch model performs classification.
6. MongoDB supports application data storage.
7. The prediction results are returned to the frontend.

## 5. Project Workflow

The audio detection process follows these steps:

1. The user uploads an audio recording.
2. The frontend sends the audio to the FastAPI backend.
3. The backend passes the audio to the prediction service.
4. Audio preprocessing converts the recording to mono, resamples it to 16 kHz and adjusts its duration to four seconds.
5. Librosa extracts a Mel spectrogram.
6. The extracted features are normalized and converted into a PyTorch tensor.
7. DeepShield V3.2 Weighted analyzes the features.
8. Softmax converts the model outputs into class probabilities.
9. The class with the highest probability is selected.
10. The prediction and confidence are returned to the frontend.

### Audio Feature Configuration

| Parameter | Configuration |
|---|---|
| Sample rate | 16,000 Hz |
| Audio channels | Mono |
| Duration | 4 seconds |
| FFT size | 1024 |
| Hop length | 256 |
| Mel bands | 128 |
| Feature shape | 128 × 251 |

## 6. Dataset Information

The training pipeline uses audio data from multiple sources.

### Dataset Sources

- **ASVspoof 2019 LA:** Research dataset containing genuine and spoofed speech, including Text-to-Speech and Voice Conversion examples.
- **Mozilla Common Voice:** Crowdsourced human speech recordings.
- **gTTS:** Used to generate additional synthetic Text-to-Speech recordings.

### Training Dataset Distribution

The recorded V3.2 training dataset contains 9,024 samples.

| Class | Samples |
|---|---:|
| Genuine | 3,608 |
| TTS | 3,608 |
| Voice Conversion | 1,808 |
| **Total** | **9,024** |

Processed audio features are represented as NumPy arrays and organized using dataset manifests.

Original datasets are not included in this repository by default. Users should obtain them from their respective sources and follow their licensing and usage requirements.

## 7. Deep Learning Model

DeepShield uses **DeepShield V3.2 Weighted**, implemented using PyTorch.

The model architecture includes:

- Convolutional layers
- Batch normalization
- SiLU activation
- Residual blocks
- Squeeze-and-Excitation channel attention
- Multi-scale feature extraction
- Average and maximum pooling
- Classification layers

Weighted CrossEntropy loss is used during training to address the class imbalance in the dataset.

### Recorded Model Performance

| Metric | Result |
|---|---:|
| Original test dataset accuracy | 98.96% |
| Best development balanced accuracy | 98.01% |
| Best checkpoint epoch | 13 |

These are recorded evaluation results on the project's original datasets. Performance on new recordings may vary.

## 8. Project Structure

The following is a simplified overview of the project:

```text
DeepShield/
│
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   └── services/
│   ├── ml/
│   ├── uploads/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── models/
├── reports/
├── datasets/
├── docs/
├── v3_1_dataset/
├── v3_1_dataset_inspection/
├── v3_2_dataset/
│
└── README.md
```

This structure is a simplified overview; individual folders may contain additional files.

## 9. Installation and Setup

### Prerequisites

- Python
- Node.js and npm
- MongoDB
- Git

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DeepShield
```

### Step 2: Backend Setup

Open a terminal:

```bash
cd backend
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

The machine-learning environment also requires compatible versions of PyTorch, Librosa and NumPy. Install the versions used by the project in the same Python environment.

Configure the required environment variables using your own local `.env` file. Never commit credentials.

### Step 3: Start the Backend

From the backend directory:

```bash
uvicorn main:app --reload
```

The FastAPI server uses the application defined in `main.py`.

### Step 4: Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite starts the React development server. The configured frontend development origin is:

```text
http://localhost:5173
```

### Step 5: Model Checkpoint

The prediction service requires the trained DeepShield V3.2 Weighted checkpoint.

Expected checkpoint filename:

```text
deepshield_v3_2_weighted_best.pth
```

Place the checkpoint in the location expected by the backend prediction service. The trained checkpoint is not included automatically unless it is explicitly distributed with the repository.

## 10. API Documentation

FastAPI provides interactive API documentation when the backend is running.

```text
http://127.0.0.1:8000/docs
```

The available API routes are organized into authentication, user, home, upload, dashboard, history, analytics and admin routers.

## 11. Model Training

The project includes a training pipeline for DeepShield V3.2 Weighted.

The training process includes:

1. Loading processed audio features.
2. Preparing class labels.
3. Creating training and development datasets.
4. Initializing the DeepShield V3 architecture.
5. Training with weighted CrossEntropy loss.
6. Updating parameters using the AdamW optimizer.
7. Evaluating model performance.
8. Saving the best model checkpoint.

## 12. Screenshots

Application screenshots can be added here to demonstrate the user interface and detection results.

Suggested screenshots:

- Login and registration
- User dashboard
- Audio upload page
- Audio prediction result
- Detection history
- Analytics dashboard
- Admin dashboard

## 13. Future Scope

- Improve generalization across different speakers and recording devices.
- Expand training datasets.
- Support additional audio formats and longer recordings.
- Improve detection of advanced synthetic speech.
- Explore real-time audio deepfake detection.
- Optimize inference performance.

## 14. Limitations

- Prediction quality can vary depending on audio quality and recording conditions.
- New voice-generation techniques may require additional training data.
- Confidence scores are model estimates and do not guarantee that a classification is correct.

## 15. Acknowledgements and References

- ASVspoof 2019 – Audio spoofing and countermeasure research dataset.
- Mozilla Common Voice – Crowdsourced speech dataset.
- gTTS – Google Text-to-Speech Python library.
- PyTorch – Deep learning framework.
- Librosa – Audio analysis library.
- FastAPI – Python web framework.
- MongoDB – Database platform.
- React – Frontend library.

---

## Author

**DeepShield Project**

An AI-powered audio deepfake detection system developed using React.js, FastAPI, PyTorch and MongoDB.
