import "../styles/HowItWorks.css";


function HowItWorks() {

  const steps = [

    {
      number: "01",
      title: "Upload Audio",
      description:
        "Select a supported WAV, MP3, M4A, FLAC or OGG audio file for analysis."
    },

    {
      number: "02",
      title: "Audio Processing",
      description:
        "DeepShield preprocesses the recording and extracts the audio features required by the model."
    },

    {
      number: "03",
      title: "AI Analysis",
      description:
        "The DeepShield V3.2 model analyzes the extracted features and evaluates the audio."
    },

    {
      number: "04",
      title: "Detection Result",
      description:
        "Receive the predicted class and confidence information showing whether the audio is Genuine, TTS or Voice Conversion."
    }

  ];


  return (

    <section className="how-it-works">


      <div className="section-title">

        <span className="how-it-works-badge">
          SIMPLE AI WORKFLOW
        </span>

        <h2>
          How DeepShield Works
        </h2>

        <p>
          Analyze an audio recording through a simple
          four-step workflow powered by deep learning.
        </p>

      </div>


      <div className="steps-container">

        {steps.map((step) => (

          <div
            className="step-card"
            key={step.number}
          >

            <div className="step-number">
              {step.number}
            </div>


            <h3>
              {step.title}
            </h3>


            <p>
              {step.description}
            </p>

          </div>

        ))}

      </div>


    </section>

  );

}


export default HowItWorks;