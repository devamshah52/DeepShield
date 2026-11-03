import MainLayout from "../layouts/MainLayout";
import "../styles/Contact.css";

import {
  FaEnvelope,
  FaMapMarkerAlt,
  FaClock,
  FaPhoneAlt,
} from "react-icons/fa";

function Contact() {
  return (
    <MainLayout>
      <section className="contact-page">

        {/* Hero */}
        <div className="contact-hero">
          <div className="contact-hero-badge">
            GET IN TOUCH
          </div>

          <h1>Contact DeepShield</h1>

          <p>
            Have questions, suggestions, or feedback?
            We would love to hear from you.
          </p>
        </div>

        {/* Contact Information */}
        <div className="contact-grid">

          <div className="contact-card">
            <div className="contact-icon">
              <FaEnvelope />
            </div>

            <h3>Email</h3>

            <p>
              deepshield.project@gmail.com
            </p>
          </div>

          <div className="contact-card">
            <div className="contact-icon">
              <FaMapMarkerAlt />
            </div>

            <h3>Location</h3>

            <p>India</p>
          </div>

          <div className="contact-card">
            <div className="contact-icon">
              <FaClock />
            </div>

            <h3>Support Hours</h3>

            <p>
              Monday - Friday
              <br />
              9:00 AM - 6:00 PM
            </p>
          </div>

          <div className="contact-card">
            <div className="contact-icon">
              <FaPhoneAlt />
            </div>

            <h3>Response Time</h3>

            <p>Within 24 Hours</p>
          </div>

        </div>

        {/* Contact Form */}
        <div className="contact-form-section">

          <div className="section-heading">
            <span>MESSAGE US</span>
            <h2>Send us a Message</h2>
            <p>
              Have a question about DeepShield? Send us a message
              and we will get back to you.
            </p>
          </div>

          <form className="contact-form">

            <div className="contact-form-row">
              <input
                type="text"
                placeholder="Full Name"
              />

              <input
                type="email"
                placeholder="Email Address"
              />
            </div>

            <input
              type="text"
              placeholder="Subject"
            />

            <textarea
              rows="6"
              placeholder="Write your message..."
            />

            <button type="submit">
              Send Message
            </button>

          </form>

        </div>

        {/* FAQ */}
        <div className="faq-section">

          <div className="section-heading">
            <span>FAQ</span>
            <h2>Frequently Asked Questions</h2>
            <p>
              Find quick answers to common questions about DeepShield.
            </p>
          </div>

          <div className="faq-card">
            <h3>What audio formats are supported?</h3>

            <p>
              DeepShield currently supports WAV, MP3, M4A, FLAC,
              and OGG audio files.
            </p>
          </div>

          <div className="faq-card">
            <h3>What types of audio can DeepShield detect?</h3>

            <p>
              DeepShield classifies audio into three categories:
              Genuine Audio, TTS Audio, and Voice Conversion Audio.
            </p>
          </div>

          <div className="faq-card">
            <h3>Does DeepShield provide a confidence score?</h3>

            <p>
              Yes. After analysis, DeepShield provides the predicted
              audio class along with its confidence score and class
              probability information.
            </p>
          </div>

          <div className="faq-card">
            <h3>Is my uploaded audio secure?</h3>

            <p>
              Audio analysis is available through authenticated
              application access and protected APIs.
            </p>
          </div>

        </div>

      </section>
    </MainLayout>
  );
}

export default Contact;