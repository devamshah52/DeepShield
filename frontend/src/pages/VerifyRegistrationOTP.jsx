import { useState, useEffect } from "react";

import {
  useLocation,
  useNavigate,
  Link
} from "react-router-dom";

import MainLayout from "../layouts/MainLayout";

import "../styles/VerifyRegistrationOTP.css";

import heroImage from "../assets/hero-ai.png";

import { FaCheckCircle } from "react-icons/fa";

import {
  verifyRegistrationOTP
} from "../api/auth";

import { toast } from "react-toastify";

function VerifyRegistrationOTP() {

  const navigate = useNavigate();

  const location = useLocation();

  const email =
    location.state?.email || "";

  const [otp, setOtp] = useState("");

  const [loading, setLoading] =
    useState(false);

  useEffect(() => {

    if (!email) {

      toast.error(
        "Please register first."
      );

      navigate("/register");

    }

  }, [email, navigate]);

  const handleVerify = async (e) => {

    e.preventDefault();

    if (otp.length !== 6) {

  toast.error(
    "Please enter a valid 6-digit OTP."
  );

  return;

}

    try {

      setLoading(true);

      await verifyRegistrationOTP(
        email,
        otp
      );

      toast.success(
        "Email verified successfully!"
      );

      setTimeout(() => {

        navigate("/login");

      }, 1200);

    } catch (error) {

      console.error(error);

      if (
        error.response?.data?.detail
      ) {

        toast.error(
          error.response.data.detail
        );

      } else {

        toast.error(
          "OTP verification failed."
        );

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
            Your Email

          </h1>

          <p>

            Enter the OTP sent to your
            registered email address to
            complete your DeepShield
            account registration.

          </p>

          <div className="verify-features">

            <div>

              <FaCheckCircle />

              <span>

                Secure Email Verification

              </span>

            </div>

            <div>

              <FaCheckCircle />

              <span>

                One-Time Password

              </span>

            </div>

            <div>

              <FaCheckCircle />

              <span>

                5 Minute Expiry

              </span>

            </div>

          </div>

        </div>

        <div className="verify-container">

          <h2>

            Verify Email

          </h2>

          <p className="verify-subtitle">

            Enter the OTP sent to

            <br />

            <strong>

              {email}

            </strong>

          </p>

          <form

            className="verify-form"

            onSubmit={handleVerify}

          >

            <div className="form-group">

              <label>

                Verification OTP

              </label>

              <div className="input-box">

                

                <input

                  type="text"

                  placeholder="Enter 6-digit OTP"

                  value={otp}

                  maxLength={6}

                  onChange={(e) =>

  setOtp(

    e.target.value.replace(/\D/g, "")

  )

}

                />

              </div>

            </div>

            <button

              type="submit"

              className="verify-btn"

              disabled={loading}

            >

              {

                loading

                  ? "Verifying..."

                  : "Verify Email"

              }

            </button>

          </form>

          <p className="register-text">

            Wrong email?

            <Link to="/register">

              {" "}Go Back

            </Link>

          </p>

        </div>

      </section>

    </MainLayout>

  );

}

export default VerifyRegistrationOTP;