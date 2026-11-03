import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";
import "../styles/Register.css";

import heroImage from "../assets/hero-ai.png";

import {
  FaEye,
  FaEyeSlash,
  FaCheckCircle
} from "react-icons/fa";

import { sendRegistrationOTP } from "../api/auth";

import { toast } from "react-toastify";


function Register() {

  const navigate = useNavigate();

  const [fullName, setFullName] = useState("");

  const [email, setEmail] = useState("");

  const [password, setPassword] = useState("");

  const [confirmPassword, setConfirmPassword] = useState("");

  const [showPassword, setShowPassword] = useState(false);

  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [loading, setLoading] = useState(false);

  const [errors, setErrors] = useState({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: ""
  });


  /* =========================
     VALIDATION
  ========================= */

  const validateName = (value) => {

    if (!value.trim()) {
      return "Full name is required.";
    }

    if (value.trim().length < 3) {
      return "Name should contain at least 3 characters.";
    }

    return "";
  };


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


  const validatePassword = (value) => {

    const passwordRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&^#])[A-Za-z\d@$!%*?&^#]{8,}$/;

    if (!value.trim()) {
      return "Password is required.";
    }

    if (!passwordRegex.test(value)) {
      return "Password must contain at least 8 characters, one uppercase letter, one lowercase letter, one number and one special character.";
    }

    return "";
  };


  const validateConfirmPassword = (value) => {

    if (!value.trim()) {
      return "Please confirm your password.";
    }

    if (value !== password) {
      return "Passwords do not match.";
    }

    return "";
  };


  /* =========================
     REGISTER
  ========================= */

  const handleRegister = async (e) => {

    e.preventDefault();

    const newErrors = {

      fullName: validateName(fullName),

      email: validateEmail(email),

      password: validatePassword(password),

      confirmPassword:
        validateConfirmPassword(confirmPassword)

    };

    setErrors(newErrors);

    if (
      newErrors.fullName ||
      newErrors.email ||
      newErrors.password ||
      newErrors.confirmPassword
    ) {
      return;
    }


    try {

      setLoading(true);

      await sendRegistrationOTP({

        full_name:
          fullName.trim(),

        email:
          email
            .trim()
            .toLowerCase(),

        password

      });


      toast.success(
        "OTP sent successfully!"
      );


      setTimeout(() => {

        navigate(
          "/verify-registration-otp",
          {
            state: {

              full_name:
                fullName.trim(),

              email:
                email
                  .trim()
                  .toLowerCase(),

              password

            }
          }
        );

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
          "Failed to send OTP."
        );

      }

    } finally {

      setLoading(false);

    }

  };


  /* =========================
     UI
  ========================= */

  return (

    <MainLayout>

      <section className="register-page">


        {/* =========================
            LEFT PANEL
        ========================= */}

        <div className="register-left">

          <img
            src={heroImage}
            alt="DeepShield"
          />


          <h1>

            Join
            <br />

            DeepShield

          </h1>


          <p>

            Create your DeepShield account and start
            analyzing audio as Genuine, TTS or Voice
            Conversion.

          </p>


          <div className="register-features">


            <div>

              <FaCheckCircle />

              <span>
                AI Audio Detection
              </span>

            </div>


            <div>

              <FaCheckCircle />

              <span>
                TTS Detection
              </span>

            </div>


            <div>

              <FaCheckCircle />

              <span>
                Voice Conversion Detection
              </span>

            </div>


            <div>

              <FaCheckCircle />

              <span>
                Secure User Authentication
              </span>

            </div>


          </div>


        </div>


        {/* =========================
            REGISTER CARD
        ========================= */}

        <div className="register-container">


          <h2>
            Create Account
          </h2>


          <p className="register-subtitle">

            Create your DeepShield account to begin
            analyzing Genuine, TTS and Voice
            Conversion audio.

          </p>


          <form
            className="register-form"
            onSubmit={handleRegister}
          >


            {/* =========================
                FULL NAME
            ========================= */}

            <div className="form-group">

              <label>
                Full Name
              </label>


              <div className="input-box">

                <input

                  type="text"

                  placeholder="Enter your full name"

                  value={fullName}

                  onChange={(e) => {

                    setFullName(
                      e.target.value
                    );

                    setErrors({

                      ...errors,

                      fullName:
                        validateName(
                          e.target.value
                        )

                    });

                  }}

                  className={
                    errors.fullName
                      ? "error-input"
                      : ""
                  }

                />

              </div>


              {errors.fullName && (

                <small className="error-text">

                  {errors.fullName}

                </small>

              )}

            </div>


            {/* =========================
                EMAIL
            ========================= */}

            <div className="form-group">

              <label>
                Email Address
              </label>


              <div className="input-box">

                <input

                  type="email"

                  placeholder="Enter your email"

                  value={email}

                  onChange={(e) => {

                    setEmail(
                      e.target.value
                    );

                    setErrors({

                      ...errors,

                      email:
                        validateEmail(
                          e.target.value
                        )

                    });

                  }}

                  className={
                    errors.email
                      ? "error-input"
                      : ""
                  }

                />

              </div>


              {errors.email && (

                <small className="error-text">

                  {errors.email}

                </small>

              )}

            </div>


            {/* =========================
                PASSWORD
            ========================= */}

            <div className="form-group">

              <label>
                Password
              </label>


              <div className="input-box password-input">


                <input

                  type={
                    showPassword
                      ? "text"
                      : "password"
                  }

                  placeholder="Create Password"

                  value={password}

                  onChange={(e) => {

                    setPassword(
                      e.target.value
                    );

                    setErrors({

                      ...errors,

                      password:
                        validatePassword(
                          e.target.value
                        )

                    });

                  }}

                  className={
                    errors.password
                      ? "error-input"
                      : ""
                  }

                />


                <button

                  type="button"

                  className="toggle-password"

                  onClick={() =>
                    setShowPassword(
                      !showPassword
                    )
                  }

                  aria-label={
                    showPassword
                      ? "Hide password"
                      : "Show password"
                  }

                >

                  {

                    showPassword

                      ? <FaEyeSlash />

                      : <FaEye />

                  }

                </button>


              </div>


              {errors.password && (

                <small className="error-text">

                  {errors.password}

                </small>

              )}

            </div>


            {/* =========================
                CONFIRM PASSWORD
            ========================= */}

            <div className="form-group">

              <label>
                Confirm Password
              </label>


              <div className="input-box password-input">


                <input

                  type={
                    showConfirmPassword
                      ? "text"
                      : "password"
                  }

                  placeholder="Confirm Password"

                  value={confirmPassword}

                  onChange={(e) => {

                    setConfirmPassword(
                      e.target.value
                    );

                    setErrors({

                      ...errors,

                      confirmPassword:
                        validateConfirmPassword(
                          e.target.value
                        )

                    });

                  }}

                  className={
                    errors.confirmPassword
                      ? "error-input"
                      : ""
                  }

                />


                <button

                  type="button"

                  className="toggle-password"

                  onClick={() =>
                    setShowConfirmPassword(
                      !showConfirmPassword
                    )
                  }

                  aria-label={
                    showConfirmPassword
                      ? "Hide confirm password"
                      : "Show confirm password"
                  }

                >

                  {

                    showConfirmPassword

                      ? <FaEyeSlash />

                      : <FaEye />

                  }

                </button>


              </div>


              {errors.confirmPassword && (

                <small className="error-text">

                  {errors.confirmPassword}

                </small>

              )}

            </div>


            {/* =========================
                REGISTER BUTTON
            ========================= */}

            <button

              type="submit"

              className="register-btn"

              disabled={loading}

            >

              {

                loading

                  ? "Sending OTP..."

                  : "Create Account"

              }

            </button>


          </form>


          {/* =========================
              LOGIN LINK
          ========================= */}

          <p className="register-text">

            Already have an account?

            <Link to="/login">

              {" "}Login

            </Link>

          </p>


        </div>


      </section>

    </MainLayout>

  );

}


export default Register;