import { useState } from "react";
import { useNavigate } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";
import "../styles/ForgotPassword.css";

import heroImage from "../assets/hero-ai.png";

import {
  FaCheckCircle
} from "react-icons/fa";

import { toast } from "react-toastify";

import { forgotPassword } from "../api/auth";

function ForgotPassword() {

  const navigate = useNavigate();

  const [email, setEmail] = useState("");

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const validateEmail = (value) => {

    const emailRegex =
      /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!value.trim()) {

      return "Email is required.";

    }

    if (!emailRegex.test(value)) {

      return "Please enter a valid email address.";

    }

    return "";

  };

  const handleSubmit = async (e) => {

    e.preventDefault();

    const emailError =
      validateEmail(email);

    setError(emailError);

    if (emailError) {

      return;

    }

    try {

      setLoading(true);

      await forgotPassword(
        email.trim().toLowerCase()
      );

      toast.success(
        "OTP sent successfully!"
      );

      setTimeout(() => {

        navigate(
          "/verify-otp",
          {
            state: {
              email:
                email
                  .trim()
                  .toLowerCase()
            }
          }
        );

      }, 1000);

    } catch (error) {

      if (
        error.response?.data?.detail
      ) {

        toast.error(
          error.response.data.detail
        );

      } else {

        toast.error(
          "Failed to send OTP."
        );

      }

    } finally {

      setLoading(false);

    }

  };

  return (

    <MainLayout>

      <section className="forgot-page">

        <div className="forgot-left">

          <img
            src={heroImage}
            alt="DeepShield"
          />

          <h1>

            Forgot
            <br />
            Password

          </h1>

          <p>

            Enter your registered email
            address and we'll send you
            a secure One-Time Password
            to reset your account.

          </p>

          <div className="forgot-features">

            <div>

              <FaCheckCircle />

              <span>

                Secure OTP Verification

              </span>

            </div>

            <div>

              <FaCheckCircle />

              <span>

                OTP Valid For 5 Minutes

              </span>

            </div>

            <div>

              <FaCheckCircle />

              <span>

                Password Encrypted

              </span>

            </div>

          </div>

        </div>

        <div className="forgot-container">

          <h2>

            Forgot Password

          </h2>

          <p className="forgot-subtitle">

            Enter your registered email
            to receive a password reset
            OTP.

          </p>

          <form
            className="forgot-form"
            onSubmit={handleSubmit}
          >

            <div className="form-group">

              <label>

                Email Address

              </label>

              <div className="input-box">

                

                <input

                  type="email"

                  placeholder="Enter your registered email"

                  value={email}

                  onChange={(e) => {

                    setEmail(
                      e.target.value
                    );

                    setError(
                      validateEmail(
                        e.target.value
                      )
                    );

                  }}

                  className={
                    error
                      ? "error-input"
                      : ""
                  }

                />

              </div>

              {

                error && (

                  <small
                    className="error-text"
                  >

                    {error}

                  </small>

                )

              }

            </div>

            <button

              type="submit"

              className="forgot-btn"

              disabled={loading}

            >

              {

                loading

                  ? "Sending OTP..."

                  : "Send OTP"

              }

            </button>

          </form>

        </div>

      </section>

    </MainLayout>

  );

}

export default ForgotPassword;