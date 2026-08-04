import MainLayout from "../layouts/MainLayout";
import "../styles/Home.css";
import heroImage from "../assets/hero-ai.png";

function Home() {
  return (
    <MainLayout>
      <section className="hero">

  <div className="hero-content">

    <h1>
      Detect Deepfake Audio with Confidence
    </h1>

    <p>
      DeepShield uses advanced Deep Learning techniques to detect
      AI-generated, cloned, and manipulated voice recordings with
      high accuracy.
    </p>

    <div className="hero-buttons">

      <button className="primary-btn">
        Analyze Audio
      </button>

      <button className="secondary-btn">
        Learn More
      </button>

    </div>

  </div>

  <div className="hero-image">
    <img src={heroImage} alt="DeepShield AI Shield" />
  </div>

</section>
    </MainLayout>
  );
}

export default Home;