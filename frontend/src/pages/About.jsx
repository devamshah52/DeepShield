import MainLayout from "../layouts/MainLayout";
import "../styles/About.css";

import {
  FaBullseye,
  FaShieldAlt,
  FaRobot,
  FaMicrophone,
  FaCheckCircle,
  FaUpload,
  FaBrain,
  FaChartLine,
  FaPython,
  FaReact,
  FaDatabase,
  FaServer,
  FaLock,
  FaCogs
} from "react-icons/fa";


function About() {

  return (

    <MainLayout>

      <section className="about-page">


        {/* =========================
            HERO
        ========================= */}

        <div className="about-hero">

          <div className="about-hero-badge">
            <FaShieldAlt />
            Deep Learning-Based Audio Detection
          </div>

          <h1>
            About DeepShield
          </h1>

          <p>
            DeepShield is a deep learning-based audio detection
            system designed to identify Genuine, TTS and Voice
            Conversion audio through AI-powered analysis.
          </p>

        </div>


        {/* =========================
            MISSION
        ========================= */}

        <div className="about-section">

          <h2>

            <FaBullseye />

            Our Mission

          </h2>

          <p>

            DeepShield is designed to help identify AI-generated
            and manipulated audio through intelligent deep learning
            analysis. The system provides a secure and user-friendly
            platform for analyzing audio and identifying whether it
            is Genuine, generated using Text-to-Speech technology,
            or produced through Voice Conversion.

          </p>

        </div>


        {/* =========================
            DETECTION TYPES
        ========================= */}

        <div className="about-section">

          <h2>

            <FaShieldAlt />

            What DeepShield Detects

          </h2>


          <div className="detect-grid">


            {/* Genuine */}

            <div className="detect-card">

              <FaCheckCircle />

              <h3>
                Genuine Audio
              </h3>

              <p>

                Authentic human speech recordings that
                are not generated or transformed using
                synthetic audio techniques.

              </p>

            </div>


            {/* TTS */}

            <div className="detect-card">

              <FaRobot />

              <h3>
                TTS Audio
              </h3>

              <p>

                Audio generated using Text-to-Speech
                systems that synthesize spoken language
                from text.

              </p>

            </div>


            {/* Voice Conversion */}

            <div className="detect-card">

              <FaMicrophone />

              <h3>
                Voice Conversion
              </h3>

              <p>

                Audio produced by transforming a speaker's
                voice characteristics while preserving
                the spoken content.

              </p>

            </div>


          </div>

        </div>


        {/* =========================
            WORKFLOW
        ========================= */}

        <div className="about-section">

          <h2>
            How DeepShield Works
          </h2>


          <div className="workflow">


            <div className="workflow-step">

              <div className="workflow-icon">
                <FaUpload />
              </div>

              <span className="workflow-number">
                01
              </span>

              <h3>
                Upload Audio
              </h3>

              <p>
                Select and upload a supported audio file.
              </p>

            </div>


            <div className="workflow-arrow">
              →
            </div>


            <div className="workflow-step">

              <div className="workflow-icon">
                <FaBrain />
              </div>

              <span className="workflow-number">
                02
              </span>

              <h3>
                AI Analysis
              </h3>

              <p>
                DeepShield extracts audio features for analysis.
              </p>

            </div>


            <div className="workflow-arrow">
              →
            </div>


            <div className="workflow-step">

              <div className="workflow-icon">
                <FaChartLine />
              </div>

              <span className="workflow-number">
                03
              </span>

              <h3>
                Prediction
              </h3>

              <p>
                The trained model predicts the audio class.
              </p>

            </div>


            <div className="workflow-arrow">
              →
            </div>


            <div className="workflow-step">

              <div className="workflow-icon">
                <FaShieldAlt />
              </div>

              <span className="workflow-number">
                04
              </span>

              <h3>
                Detection Result
              </h3>

              <p>
                View the predicted class and confidence.
              </p>

            </div>


          </div>

        </div>


        {/* =========================
            TECHNOLOGY
        ========================= */}

        <div className="about-section">

          <h2>
            Technology Stack
          </h2>


          <div className="tech-grid">


            <div>

              <FaReact />

              <span>
                React
              </span>

            </div>


            <div>

              <FaPython />

              <span>
                Python
              </span>

            </div>


            <div>

              <FaServer />

              <span>
                FastAPI
              </span>

            </div>


            <div>

              <FaDatabase />

              <span>
                MongoDB
              </span>

            </div>


            <div>

              <FaLock />

              <span>
                JWT Authentication
              </span>

            </div>


            <div>

              <FaBrain />

              <span>
                PyTorch
              </span>

            </div>


            <div>

              <FaCogs />

              <span>
                Deep Learning
              </span>

            </div>


          </div>

        </div>


        {/* =========================
            CURRENT SYSTEM
        ========================= */}

        <div className="about-section">

          <h2>
            Current System
          </h2>


          <div className="status-grid">


            <div className="status-card">

              <FaLock />

              <div>

                <h3>
                  Authentication
                </h3>

                <span className="status-complete">
                  Completed
                </span>

              </div>

            </div>


            <div className="status-card">

              <FaUpload />

              <div>

                <h3>
                  Audio Upload
                </h3>

                <span className="status-complete">
                  Completed
                </span>

              </div>

            </div>


            <div className="status-card">

              <FaBrain />

              <div>

                <h3>
                  AI Detection
                </h3>

                <span className="status-complete">
                  Completed
                </span>

              </div>

            </div>


            <div className="status-card">

              <FaChartLine />

              <div>

                <h3>
                  Analytics
                </h3>

                <span className="status-complete">
                  Completed
                </span>

              </div>

            </div>


            <div className="status-card">

              <FaShieldAlt />

              <div>

                <h3>
                  Detection History
                </h3>

                <span className="status-complete">
                  Completed
                </span>

              </div>

            </div>


            <div className="status-card">

              <FaCogs />

              <div>

                <h3>
                  User Profile
                </h3>

                <span className="status-complete">
                  Completed
                </span>

              </div>

            </div>


          </div>

        </div>


        {/* =========================
            MODEL
        ========================= */}

        <div className="about-section about-model-section">

          <div className="model-content">

            <div className="model-icon">
              <FaBrain />
            </div>

            <div>

              <h2>
                DeepShield V3.2
              </h2>

              <p>

                DeepShield uses a trained deep learning classification
                model to analyze audio features and classify recordings
                into three categories: Genuine, TTS and Voice Conversion.

              </p>

            </div>

          </div>


          <div className="model-stats">

            <div>

              <strong>
                3
              </strong>

              <span>
                Audio Classes
              </span>

            </div>


            <div>

              <strong>
                128 × 251
              </strong>

              <span>
                Feature Input
              </span>

            </div>


            <div>

              <strong>
                98.96%
              </strong>

              <span>
                Test Accuracy
              </span>

            </div>

          </div>

        </div>


      </section>

    </MainLayout>

  );

}


export default About;