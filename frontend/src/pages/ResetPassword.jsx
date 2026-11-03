import { useState } from "react";
import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";
import "../styles/ResetPassword.css";

import heroImage from "../assets/hero-ai.png";

import {
  FaLock,
  FaEye,
  FaEyeSlash,
  FaCheckCircle
} from "react-icons/fa";

import { toast } from "react-toastify";

import { resetPassword } from "../api/auth";

function ResetPassword() {

  const navigate = useNavigate();

  const location = useLocation();

  const email = location.state?.email || "";

  const otp = location.state?.otp || "";

  useEffect(() => {

  if (!email || !otp) {

    toast.error("Please verify your OTP first.");

    navigate("/forgot-password");

  }

}, [email, otp, navigate]);

  const [password, setPassword] = useState("");

  const [confirmPassword, setConfirmPassword] = useState("");

  const [showPassword, setShowPassword] = useState(false);

  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [loading, setLoading] = useState(false);

  const [errors, setErrors] = useState({

    password: "",

    confirmPassword: ""

  });

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

  const handleSubmit = async (e) => {

    e.preventDefault();

    const passwordError = validatePassword(password);

    const confirmPasswordError =
      validateConfirmPassword(confirmPassword);

    setErrors({

      password: passwordError,

      confirmPassword: confirmPasswordError

    });

    if (

      passwordError ||

      confirmPasswordError

    ) {

      return;

    }

    try {

      setLoading(true);

      await resetPassword(

        email,

        otp,

        password

      );

      toast.success(

        "Password Reset Successful!"

      );

      setTimeout(() => {

        navigate("/login");

      }, 1200);

    } catch (error) {

      if (

        error.response?.data?.detail

      ) {

        toast.error(

          error.response.data.detail

        );

      } else {

        toast.error(

          "Password Reset Failed."

        );

      }

    } finally {

      setLoading(false);

    }

  };

  return (

    <MainLayout>

      <section className="reset-page">

        <div className="reset-left">

          <img
            src={heroImage}
            alt="DeepShield"
          />

          <h1>

            Reset
            <br />
            Password

          </h1>

          <p>

            Create a new secure password
            for your DeepShield account.

          </p>

          <div className="reset-features">

            <div>

              <FaCheckCircle />

              <span>

                Strong Password

              </span>

            </div>

            <div>

              <FaCheckCircle />

              <span>

                Secure Encryption

              </span>

            </div>

            <div>

              <FaCheckCircle />

              <span>

                Instant Password Update

              </span>

            </div>

          </div>

        </div>

        <div className="reset-container">

          <h2>

            Create New Password

          </h2>

          <p className="reset-subtitle">

            Your new password must be
            strong and secure.

          </p>

          <form

            className="reset-form"

            onSubmit={handleSubmit}

          >

            <div className="form-group">

              <label>

                New Password

              </label>

              <div className="input-box password-input">

                <FaLock className="input-icon" />

                <input

                  type={

                    showPassword

                      ? "text"

                      : "password"

                  }

                  placeholder="New Password"

                  value={password}

                  onChange={(e) => {

                    setPassword(e.target.value);

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

                >

                  {

                    showPassword

                      ? <FaEyeSlash />

                      : <FaEye />

                  }

                </button>

              </div>

              {

                errors.password && (

                  <small className="error-text">

                    {errors.password}

                  </small>

                )

              }

            </div>

            <div className="form-group">

              <label>

                Confirm Password

              </label>

              <div className="input-box password-input">

                <FaLock className="input-icon" />

                <input

                  type={

                    showConfirmPassword

                      ? "text"

                      : "password"

                  }

                  placeholder="Confirm Password"

                  value={confirmPassword}

                  onChange={(e) => {

                    setConfirmPassword(e.target.value);

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

                >

                  {

                    showConfirmPassword

                      ? <FaEyeSlash />

                      : <FaEye />

                  }

                </button>

              </div>

              {

                errors.confirmPassword && (

                  <small className="error-text">

                    {errors.confirmPassword}

                  </small>

                )

              }

            </div>

            <button

              type="submit"

              className="reset-btn"

              disabled={loading}

            >

              {

                loading

                  ? "Updating Password..."

                  : "Reset Password"

              }

            </button>

          </form>

        </div>

      </section>

    </MainLayout>

  );

}

export default ResetPassword;