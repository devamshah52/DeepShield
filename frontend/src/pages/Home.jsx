import { Link } from "react-router-dom";
import MainLayout from "../layouts/MainLayout";
import "../styles/Home.css";

import heroImage from "../assets/hero-ai.png";

import Features from "../components/Features";
import HowItWorks from "../components/HowItWorks";

function Home() {

  return (

    <MainLayout>

      <section className="hero">

        <div className="hero-content">

          <span className="hero-badge">
            AI-Powered Audio Detection
          </span>


          <h1>
            Detect Genuine
            <br />
            TTS &
            <br />
            Voice Conversion Audio
          </h1>


          <p>
            DeepShield uses deep learning to analyze voice recordings
            and classify audio as Genuine, TTS-generated, or
            Voice Conversion audio with a prediction and confidence score.
          </p>


          <div className="hero-buttons">

            <Link
              to="/login"
              className="primary-btn"
            >
              Analyze Audio
            </Link>


            <a
              href="#features"
              className="secondary-btn"
            >
              Explore Features
            </a>

          </div>


          <div className="hero-stats">

            <div className="stat-card">

              <h3>98.96%</h3>

              <span>
                V3.2 Test Accuracy
              </span>

            </div>


            <div className="stat-card">

              <h3>3</h3>

              <span>
                Audio Classes
              </span>

            </div>


            <div className="stat-card">

              <h3>AI</h3>

              <span>
                Deep Learning Detection
              </span>

            </div>


            <div className="stat-card">

              <h3>Secure</h3>

              <span>
                Audio Analysis
              </span>

            </div>

          </div>

        </div>


        <div className="hero-image">

          <img
            src={heroImage}
            alt="DeepShield AI audio detection"
          />

        </div>

      </section>


      <div id="features">

        <Features />

      </div>


      <HowItWorks />

    </MainLayout>

  );

}

export default Home;