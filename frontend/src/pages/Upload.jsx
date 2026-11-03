import { useState, useRef, useEffect } from "react";

import MainLayout from "../layouts/MainLayout";

import "../styles/Upload.css";

import { toast } from "react-toastify";

import { uploadAudio } from "../api/auth";

import {
  getPredictionClass,
  getStatusClass,
  formatStatus,
} from "../utils/detection";


function Upload() {

  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);

  const [audioUrl, setAudioUrl] = useState(null);

  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState(null);

  const [dragActive, setDragActive] = useState(false);


  // ==========================================================
  // Supported formats
  // ==========================================================

  const SUPPORTED_EXTENSIONS = [
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
  ];


  // ==========================================================
  // Audio preview URL
  //
  // Creates a temporary browser URL for the selected
  // audio file so the user can play it before analysis.
  // ==========================================================

  useEffect(() => {

    if (!selectedFile) {

      setAudioUrl(null);

      return;

    }


    const url = URL.createObjectURL(
      selectedFile
    );

    setAudioUrl(url);


    // --------------------------------------------------------
    // Release the temporary browser URL when the file changes
    // --------------------------------------------------------

    return () => {

      URL.revokeObjectURL(url);

    };

  }, [selectedFile]);


  // ==========================================================
  // Choose file
  // ==========================================================

  const chooseFile = () => {

    if (!loading) {

      fileInputRef.current?.click();

    }

  };


  // ==========================================================
  // Validate file
  // ==========================================================

  const isSupportedFile = (file) => {

    if (!file?.name) {

      return false;

    }


    const fileName =
      file.name.toLowerCase();


    return SUPPORTED_EXTENSIONS.some(
      (extension) =>
        fileName.endsWith(extension)
    );

  };


  // ==========================================================
  // Process selected file
  // ==========================================================

  const processSelectedFile = (file) => {

    if (!file) {

      return;

    }


    if (!isSupportedFile(file)) {

      toast.error(
        "Unsupported format. Please choose WAV, MP3, M4A, FLAC, or OGG."
      );

      return;

    }


    setSelectedFile(file);

    setResult(null);

  };


  // ==========================================================
  // File input change
  // ==========================================================

  const handleFileChange = (event) => {

    const file =
      event.target.files?.[0];


    processSelectedFile(file);

  };


  // ==========================================================
  // Drag events
  // ==========================================================

  const handleDragEnter = (event) => {

    event.preventDefault();

    event.stopPropagation();


    if (!loading) {

      setDragActive(true);

    }

  };


  const handleDragOver = (event) => {

    event.preventDefault();

    event.stopPropagation();


    if (!loading) {

      setDragActive(true);

    }

  };


  const handleDragLeave = (event) => {

    event.preventDefault();

    event.stopPropagation();

    setDragActive(false);

  };


  const handleDrop = (event) => {

    event.preventDefault();

    event.stopPropagation();


    setDragActive(false);


    if (loading) {

      return;

    }


    const file =
      event.dataTransfer.files?.[0];


    processSelectedFile(file);

  };


  // ==========================================================
  // Remove selected file
  // ==========================================================

  const removeSelectedFile = () => {

    if (loading) {

      return;

    }


    setSelectedFile(null);

    setAudioUrl(null);

    setResult(null);


    if (fileInputRef.current) {

      fileInputRef.current.value = "";

    }

  };


  // ==========================================================
  // Format file size
  // ==========================================================

  const formatFileSize = (bytes) => {

    if (!bytes) {

      return "0 Bytes";

    }


    const units = [
      "Bytes",
      "KB",
      "MB",
      "GB",
    ];


    const index = Math.floor(
      Math.log(bytes) / Math.log(1024)
    );


    return (
      `${parseFloat(
        (
          bytes /
          Math.pow(1024, index)
        ).toFixed(2)
      )} ${units[index]}`
    );

  };


  // ==========================================================
  // Upload and analyze
  // ==========================================================

  const handleUpload = async () => {

    if (!selectedFile) {

      toast.error(
        "Please select an audio file."
      );

      return;

    }


    try {

      setLoading(true);

      setResult(null);


      const response =
        await uploadAudio(
          selectedFile
        );


      const data =
        response.data;


      setResult(data);


      toast.success(
        data.message ||
        "Audio analyzed successfully."
      );


      // ------------------------------------------------------
      // Clear selected file after successful analysis
      // ------------------------------------------------------

      setSelectedFile(null);

      setAudioUrl(null);


      if (fileInputRef.current) {

        fileInputRef.current.value = "";

      }


    } catch (error) {

      console.error(error);


      const message =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        "Audio analysis failed.";


      toast.error(message);

    } finally {

      setLoading(false);

    }

  };


  // ==========================================================
  // Render
  // ==========================================================

  return (

    <MainLayout>

      <section className="upload-page">

        <div className="upload-card">


          {/* ================================================== */}
          {/* HEADER */}
          {/* ================================================== */}

          <h1>
            🎵 Upload Audio
          </h1>


          <p>
            Upload an audio file for
            DeepShield AI analysis.
          </p>


          <p className="supported-formats">

            Supported formats:

            <strong>
              WAV • MP3 • M4A • FLAC • OGG
            </strong>

          </p>


          {/* ================================================== */}
          {/* UPLOAD AREA */}
          {/* ================================================== */}

          <div
            className={`upload-box ${
              dragActive
                ? "drag-active"
                : ""
            }`}

            onDragEnter={handleDragEnter}

            onDragOver={handleDragOver}

            onDragLeave={handleDragLeave}

            onDrop={handleDrop}

          >


            <div className="upload-icon">

              🎵

            </div>


            <p>

              {dragActive

                ? "Drop your audio file here"

                : "Drag & Drop your audio file here"

              }

            </p>


            <span>

              or choose a file from your computer

            </span>


            <input

              type="file"

              accept=".wav,.mp3,.m4a,.flac,.ogg"

              ref={fileInputRef}

              onChange={handleFileChange}

              hidden

            />


            <button

              type="button"

              onClick={chooseFile}

              disabled={loading}

            >

              Choose Audio File

            </button>


            {/* ================================================== */}
            {/* SELECTED FILE */}
            {/* ================================================== */}

            {selectedFile && (

              <div className="selected-file">


                <div className="selected-file-info">

                  <div className="selected-file-icon">

                    🎧

                  </div>


                  <div>

                    <strong>

                      {selectedFile.name}

                    </strong>


                    <small>

                      {formatFileSize(
                        selectedFile.size
                      )}

                    </small>

                  </div>

                </div>


                {!loading && (

                  <button

                    type="button"

                    className="remove-file-btn"

                    onClick={
                      removeSelectedFile
                    }

                  >

                    ✕

                  </button>

                )}

              </div>

            )}


            {/* ================================================== */}
            {/* AUDIO PREVIEW */}
            {/* ================================================== */}

            {selectedFile && audioUrl && (

              <div className="audio-preview">

                <div className="audio-preview-header">

                  <div className="audio-preview-title">

                    <span className="audio-preview-icon">
                      ▶
                    </span>

                    <div>

                      <strong>
                        Audio Preview
                      </strong>

                      <small>
                        Listen before analysis
                      </small>

                    </div>

                  </div>

                </div>


                <audio
                  key={audioUrl}
                  className="audio-player"
                  controls
                  preload="metadata"
                  src={audioUrl}
                >

                  Your browser does not support
                  audio playback.

                </audio>

              </div>

            )}


            {/* ================================================== */}
            {/* ANALYZE BUTTON */}
            {/* ================================================== */}

            <button

              type="button"

              className="upload-btn"

              onClick={handleUpload}

              disabled={
                loading ||
                !selectedFile
              }

            >

              {loading

                ? "🔄 Analyzing..."

                : "🔍 Analyze Audio"

              }

            </button>


            {loading && (

              <div className="analysis-loading">

                <div className="loading-spinner"></div>


                <p>

                  DeepShield AI is analyzing your audio...

                </p>


                <small>

                  Please wait. Do not close this page.

                </small>

              </div>

            )}

          </div>


                    {/* ================================================== */}
          {/* ANALYSIS RESULT */}
          {/* ================================================== */}

          {result && (

            <div className="prediction-result">

              {/* ================================================== */}
              {/* RESULT HEADER */}
              {/* ================================================== */}

              <div className="result-header">

                <div>

                  <span className="result-eyebrow">
                    DEEPSHIELD AI ANALYSIS
                  </span>

                  <h2>
                    Analysis Result
                  </h2>

                </div>

                <div className="result-complete-icon">
                  ✓
                </div>

              </div>


              {/* ================================================== */}
              {/* MAIN PREDICTION */}
              {/* ================================================== */}

              <div className="prediction-hero">

                <div className="prediction-hero-label">

                  <span className="prediction-dot"></span>

                  <span>
                    Detected Audio Type
                  </span>

                </div>


                <div
                  className={`prediction-hero-value ${
                    getPredictionClass(
                      result.prediction
                    )
                  }`}
                >

                  {result.prediction || "Pending"}

                </div>


                <p className="prediction-description">

                  {result.prediction === "Genuine" && (
                    <>
                      The audio appears to be
                      <strong> genuine human speech </strong>
                      based on the DeepShield analysis.
                    </>
                  )}

                  {result.prediction === "TTS" && (
                    <>
                      The audio shows characteristics
                      associated with
                      <strong> AI-generated Text-to-Speech </strong>
                      audio.
                    </>
                  )}

                  {result.prediction === "Voice Conversion" && (
                    <>
                      The audio shows characteristics
                      associated with
                      <strong> AI-based Voice Conversion </strong>
                      or voice transformation.
                    </>
                  )}

                  {![
                    "Genuine",
                    "TTS",
                    "Voice Conversion",
                  ].includes(result.prediction) && (
                    <>
                      The audio analysis has been completed.
                    </>
                  )}

                </p>

              </div>


              {/* ================================================== */}
              {/* CONFIDENCE */}
              {/* ================================================== */}

              <div className="confidence-card">

                <div className="confidence-header">

                  <div>

                    <span className="confidence-label">
                      Detection Confidence
                    </span>

                    <small>
                      Model confidence in the predicted class
                    </small>

                  </div>


                  <strong className="confidence-percentage">

                    {Number(
                      result.confidence ?? 0
                    ).toFixed(2)}

                    %

                  </strong>

                </div>


                <div className="confidence-track">

                  <div
                    className="confidence-fill"
                    style={{
                      width: `${Math.min(
                        Math.max(
                          Number(
                            result.confidence ?? 0
                          ),
                          0
                        ),
                        100
                      )}%`,
                    }}
                  />

                </div>

              </div>


              {/* ================================================== */}
              {/* CLASS PROBABILITIES */}
              {/* ================================================== */}

              {result.probabilities && (

                <div className="probabilities">

                  <div className="section-heading">

                    <div>

                      <h3>
                        Class Probabilities
                      </h3>

                      <span>
                        Probability distribution across all classes
                      </span>

                    </div>

                  </div>


                  {/* Genuine */}

                  <div className="probability-card">

                    <div className="probability-top">

                      <div className="probability-name">

                        <span className="probability-indicator genuine-indicator"></span>

                        <span>
                          Genuine
                        </span>

                      </div>

                      <strong>
                        {Number(
                          result.probabilities.Genuine ?? 0
                        ).toFixed(2)}
                        %
                      </strong>

                    </div>


                    <div className="probability-track">

                      <div
                        className="probability-fill genuine-fill"
                        style={{
                          width: `${Math.min(
                            Math.max(
                              Number(
                                result.probabilities.Genuine ?? 0
                              ),
                              0
                            ),
                            100
                          )}%`,
                        }}
                      />

                    </div>

                  </div>


                  {/* TTS */}

                  <div className="probability-card">

                    <div className="probability-top">

                      <div className="probability-name">

                        <span className="probability-indicator tts-indicator"></span>

                        <span>
                          TTS
                        </span>

                      </div>

                      <strong>
                        {Number(
                          result.probabilities.TTS ?? 0
                        ).toFixed(2)}
                        %
                      </strong>

                    </div>


                    <div className="probability-track">

                      <div
                        className="probability-fill tts-fill"
                        style={{
                          width: `${Math.min(
                            Math.max(
                              Number(
                                result.probabilities.TTS ?? 0
                              ),
                              0
                            ),
                            100
                          )}%`,
                        }}
                      />

                    </div>

                  </div>


                  {/* Voice Conversion */}

                  <div className="probability-card">

                    <div className="probability-top">

                      <div className="probability-name">

                        <span className="probability-indicator vc-indicator"></span>

                        <span>
                          Voice Conversion
                        </span>

                      </div>

                      <strong>
                        {Number(
                          result.probabilities[
                            "Voice Conversion"
                          ] ?? 0
                        ).toFixed(2)}
                        %
                      </strong>

                    </div>


                    <div className="probability-track">

                      <div
                        className="probability-fill vc-fill"
                        style={{
                          width: `${Math.min(
                            Math.max(
                              Number(
                                result.probabilities[
                                  "Voice Conversion"
                                ] ?? 0
                              ),
                              0
                            ),
                            100
                          )}%`,
                        }}
                      />

                    </div>

                  </div>

                </div>

              )}


              {/* ================================================== */}
              {/* ANALYSIS INFORMATION */}
              {/* ================================================== */}

              <div className="analysis-info">

                <div>

                  <span>
                    Status
                  </span>

                  <span
                    className={getStatusClass(
                      result.status || "completed"
                    )}
                  >

                    {formatStatus(
                      result.status || "completed"
                    )}

                  </span>

                </div>


                {result.original_filename && (

                  <div>

                    <span>
                      File
                    </span>

                    <strong>
                      {result.original_filename}
                    </strong>

                  </div>

                )}


                {result.model && (

                  <div>

                    <span>
                      Detection Model
                    </span>

                    <strong>
                      DeepShield V3.2 Weighted
                    </strong>

                  </div>

                )}


                {result.feature_shape && (

                  <div>

                    <span>
                      Feature Shape
                    </span>

                    <strong>
                      {result.feature_shape.join(" × ")}
                    </strong>

                  </div>

                )}

              </div>


              {/* ================================================== */}
              {/* NEW ANALYSIS */}
              {/* ================================================== */}

              <button

                type="button"

                className="new-analysis-btn"

                onClick={() => {

                  setResult(null);

                }}

              >

                🔄 Analyze Another Audio

              </button>

            </div>

          )}

        </div>

      </section>

    </MainLayout>

  );

}


export default Upload;