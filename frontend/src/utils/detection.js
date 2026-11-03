// ============================================================
// DEEPSHIELD DETECTION HELPERS
// ============================================================


// ============================================================
// PREDICTION CLASS
// ============================================================

export const getPredictionClass = (prediction) => {

  switch (prediction) {

    case "Genuine":
      return "prediction-badge prediction-genuine";

    case "TTS":
      return "prediction-badge prediction-tts";

    case "Voice Conversion":
      return "prediction-badge prediction-voice-conversion";

    default:
      return "prediction-badge prediction-pending";

  }

};


// ============================================================
// STATUS CLASS
// ============================================================

export const getStatusClass = (status) => {

  switch (status) {

    case "completed":
      return "status-badge status-completed";

    case "uploaded":
      return "status-badge status-pending";

    case "processing":
      return "status-badge status-processing";

    case "failed":
      return "status-badge status-failed";

    default:
      return "status-badge status-pending";

  }

};


// ============================================================
// FORMAT PREDICTION
// ============================================================

export const formatPrediction = (prediction) => {

  if (!prediction) {

    return "Pending";

  }

  return prediction;

};


// ============================================================
// FORMAT STATUS
// ============================================================

export const formatStatus = (status) => {

  switch (status) {

    case "completed":
      return "Completed";

    case "uploaded":
      return "Waiting for analysis";

    case "processing":
      return "Analyzing";

    case "failed":
      return "Failed";

    default:
      return status || "Pending";

  }

};