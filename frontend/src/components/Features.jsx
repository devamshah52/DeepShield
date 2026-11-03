import "../styles/Features.css";

import {
  FaShieldAlt,
  FaRobot,
  FaMicrophoneAlt,
  FaChartBar,
  FaHistory,
  FaLock
} from "react-icons/fa";


const features = [

  {
    icon: <FaShieldAlt />,
    title: "AI Audio Detection",
    description:
      "Analyze audio using deep learning to classify recordings as Genuine, TTS or Voice Conversion."
  },

  {
    icon: <FaRobot />,
    title: "TTS Detection",
    description:
      "Identify audio generated using Text-to-Speech systems through learned audio characteristics."
  },

  {
    icon: <FaMicrophoneAlt />,
    title: "Voice Conversion Detection",
    description:
      "Detect audio produced by transforming the characteristics of a speaker's voice."
  },

  {
    icon: <FaChartBar />,
    title: "Confidence Analysis",
    description:
      "View the model's confidence and class probabilities for every audio prediction."
  },

  {
    icon: <FaHistory />,
    title: "Detection History",
    description:
      "Review previous audio analyses, predictions and confidence results from your account."
  },

  {
    icon: <FaLock />,
    title: "Secure Platform",
    description:
      "Use authenticated access to keep your audio analysis experience secure and organized."
  }

];


function Features() {

  return (

    <section className="features">

      <div className="section-title">

        <span className="features-badge">
          DEEPSHIELD CAPABILITIES
        </span>

        <h2>
          Powerful Audio Detection
        </h2>

        <p>
          DeepShield combines deep learning analysis,
          confidence scoring and secure user features
          to help identify Genuine, TTS and Voice
          Conversion audio.
        </p>

      </div>


      <div className="feature-grid">

        {features.map((feature, index) => (

          <div
            className="feature-card"
            key={index}
          >

            <div className="feature-icon">

              {feature.icon}

            </div>


            <div className="feature-content">

              <h3>
                {feature.title}
              </h3>

              <p>
                {feature.description}
              </p>

            </div>

          </div>

        ))}

      </div>

    </section>

  );

}


export default Features;