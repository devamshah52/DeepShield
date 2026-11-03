import "../styles/Footer.css";

function Footer() {

  return (

    <footer className="footer">

      <div className="footer-container">

        <div className="footer-brand">

          <h3>DeepShield</h3>

          <p>
            Deep Learning-Based Deepfake Audio
            and Voice Clone Detection System.
          </p>

        </div>


        <div className="footer-links">

          <span>
            AI-Powered Audio Detection
          </span>

          <span>
            •
          </span>

          <span>
            Secure & Reliable Analysis
          </span>

        </div>


        <div className="footer-bottom">

          <span>
            © {new Date().getFullYear()} DeepShield
          </span>

          <span>
            All rights reserved.
          </span>

        </div>

      </div>

    </footer>

  );

}

export default Footer;