import { useState } from "react";
import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";



import MainLayout from "../layouts/MainLayout";
import "../styles/VerifyOTP.css";

import heroImage from "../assets/hero-ai.png";

import { FaKey, FaCheckCircle } from "react-icons/fa";

import { toast } from "react-toastify";

import { verifyOTP } from "../api/auth";



function VerifyOTP() {

  const navigate = useNavigate();

  const location = useLocation();

  const email = location.state?.email || "";

  useEffect(() => {

  if (!email) {

    toast.error("Please start the password reset process first.");

    navigate("/forgot-password");

  }

}, [email, navigate]);

  const [otp, setOtp] = useState("");

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const validateOTP = (value) => {

    if (!value.trim()) {

      return "OTP is required.";

    }

    if (!/^\d{6}$/.test(value)) {

      return "OTP must be exactly 6 digits.";

    }

    return "";

  };

  const handleSubmit = async (e) => {

    e.preventDefault();

    const otpError = validateOTP(otp);

    setError(otpError);

    if (otpError) {

      return;

    }

    try {

      setLoading(true);

      await verifyOTP(email, otp);

      toast.success("OTP Verified Successfully!");

      setTimeout(() => {

        navigate("/reset-password", {

          state: {

            email,

            otp

          }

        });

      }, 1000);

    } catch (error) {

      if (error.response?.data?.detail) {

        toast.error(error.response.data.detail);

      } else {

        toast.error("OTP Verification Failed.");

      }

    } finally {

      setLoading(false);

    }

  };

  return (

    <MainLayout>

      <section className="verify-page">

        <div className="verify-left">

          <img

            src={heroImage}

            alt="DeepShield"

          />

          <h1>

            Verify
            <br />
            OTP

          </h1>

          <p>

            Enter the 6-digit OTP sent to your
            registered email address.

          </p>

          <div className="verify-features">

            <div>

              <FaCheckCircle />

              <span>

                Secure Verification

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

                One-Time Use OTP

              </span>

            </div>

          </div>

        </div>

        <div className="verify-container">

          <h2>

            Verify OTP

          </h2>

          <p className="verify-subtitle">

            Enter the OTP received in your email.

          </p>

          <form

            className="verify-form"

            onSubmit={handleSubmit}

          >

            <div className="form-group">

              <label>

                One-Time Password

              </label>

              <div className="input-box">

                <FaKey className="input-icon" />

                <input

                  type="text"

                  maxLength="6"

                  placeholder="Enter 6-digit OTP"

                  value={otp}

                  onChange={(e) => {

                    setOtp(e.target.value);

                    setError(

                      validateOTP(

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

                  <small className="error-text">

                    {error}

                  </small>

                )

              }

            </div>

            <button

              type="submit"

              className="verify-btn"

              disabled={loading}

            >

              {

                loading

                  ? "Verifying..."

                  : "Verify OTP"

              }

            </button>

          </form>

        </div>

      </section>

    </MainLayout>

  );

}

export default VerifyOTP;