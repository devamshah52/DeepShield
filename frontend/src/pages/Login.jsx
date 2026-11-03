import { useState } from "react";

import { Link, useNavigate } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";

import "../styles/Login.css";

import heroImage from "../assets/hero-ai.png";

import { toast } from "react-toastify";

import { useUser } from "../context/UserContext";

import {
  FaEye,
  FaEyeSlash,
  FaCheckCircle,
  FaEnvelope,
  FaLock
} from "react-icons/fa";

import { loginUser } from "../api/auth";


function Login() {

  const navigate = useNavigate();

  const { setUser } = useUser();


  const [email, setEmail] = useState(
    localStorage.getItem("savedEmail") || ""
  );


  const [password, setPassword] = useState("");

  const [showPassword, setShowPassword] =
    useState(false);


  const [rememberMe, setRememberMe] =
    useState(
      !!localStorage.getItem("savedEmail")
    );


  const [loading, setLoading] =
    useState(false);


  const [errors, setErrors] = useState({
    email: "",
    password: ""
  });


  // ============================================================
  // EMAIL VALIDATION
  // ============================================================

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


  // ============================================================
  // PASSWORD VALIDATION
  // ============================================================

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


  // ============================================================
  // HANDLE LOGIN
  // ============================================================

  const handleLogin = async (e) => {

    e.preventDefault();


    const emailError =
      validateEmail(email);


    const passwordError =
      validatePassword(password);


    setErrors({

      email: emailError,

      password: passwordError

    });


    if (
      emailError ||
      passwordError
    ) {

      return;

    }


    try {

      setLoading(true);


      // ======================================================
      // LOGIN
      // ======================================================

      const response = await loginUser({

        email:
          email.trim().toLowerCase(),

        password

      });


      // ======================================================
      // CHECK ACCESS TOKEN
      // ======================================================

      if (!response.data.access_token) {

        toast.error(
          "Login failed."
        );

        return;

      }


      localStorage.setItem(

        "token",

        response.data.access_token

      );


      // ======================================================
      // REMEMBER ME
      // ======================================================

      if (rememberMe) {

        localStorage.setItem(

          "savedEmail",

          email.trim().toLowerCase()

        );

      } else {

        localStorage.removeItem(
          "savedEmail"
        );

      }


      // ======================================================
      // GET USER DASHBOARD DATA
      // ======================================================

      const userResponse = await fetch(

        "http://127.0.0.1:8000/dashboard",

        {

          headers: {

            Authorization:
              `Bearer ${response.data.access_token}`,

          },

        }

      );


      const userData =
        await userResponse.json();


      setUser(userData);


      toast.success(
        "Login Successful!"
      );


      setTimeout(() => {

        navigate("/dashboard");

      }, 800);


    } catch (error) {

      console.error(
        "Login error:",
        error
      );


      // ======================================================
      // BACKEND ERROR HANDLING
      // ======================================================

      const statusCode =
        error?.response?.status;


      const backendMessage =
        error?.response?.data?.detail;


      // ------------------------------------------------------
      // BLOCKED USER
      // ------------------------------------------------------

      if (
        statusCode === 403
      ) {

        setErrors({

          email: "",

          password: ""

        });


        toast.error(

          backendMessage ||
          "Your account has been blocked by the administrator."

        );


        return;

      }


      // ------------------------------------------------------
      // INVALID EMAIL / PASSWORD
      // ------------------------------------------------------

      if (
        statusCode === 401
      ) {

        setErrors({

          email: "",

          password:
            backendMessage ||
            "Invalid email or password."

        });


        toast.error(

          backendMessage ||
          "Invalid email or password."

        );


        return;

      }


      // ------------------------------------------------------
      // OTHER BACKEND ERRORS
      // ------------------------------------------------------

      if (backendMessage) {

        setErrors({

          email: "",

          password: backendMessage

        });


        toast.error(
          backendMessage
        );


        return;

      }


      // ------------------------------------------------------
      // NETWORK / UNKNOWN ERROR
      // ------------------------------------------------------

      setErrors({

        email: "",

        password:
          "Unable to connect to the server."

      });


      toast.error(
        "Unable to connect to the server."
      );

    }

    finally {

      setLoading(false);

    }

  };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <MainLayout>

      <section className="login-page">


        {/* ================================================== */}
        {/* LEFT PANEL */}
        {/* ================================================== */}

        <div className="login-left">

          <img
            src={heroImage}
            alt="DeepShield AI"
          />


          <h1>

            Welcome to

            <br />

            DeepShield

          </h1>


          <p>

            AI-powered platform for detecting Genuine,
            TTS and Voice Conversion audio using Deep Learning.

          </p>


          <div className="login-features">


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


          </div>

        </div>


        {/* ================================================== */}
        {/* LOGIN CONTAINER */}
        {/* ================================================== */}

        <div className="login-container">


          <h2>

            Welcome Back

          </h2>


          <p className="login-subtitle">

            Sign in to continue to
            DeepShield.

          </p>


          <form

            className="login-form"

            onSubmit={handleLogin}

          >


            {/* ============================================ */}
            {/* EMAIL */}
            {/* ============================================ */}

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


            {/* ============================================ */}
            {/* PASSWORD */}
            {/* ============================================ */}

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

                  placeholder="Enter your password"

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


            {/* ============================================ */}
            {/* LOGIN OPTIONS */}
            {/* ============================================ */}

            <div className="login-options">


              <label className="remember-me">

                <input

                  type="checkbox"

                  checked={rememberMe}

                  onChange={() =>

                    setRememberMe(

                      !rememberMe

                    )

                  }

                />

                Remember Me

              </label>


              <Link

                to="/forgot-password"

                className="forgot-password"

              >

                Forgot Password?

              </Link>


            </div>


            {/* ============================================ */}
            {/* LOGIN BUTTON */}
            {/* ============================================ */}

            <button

              type="submit"

              className="login-btn"

              disabled={loading}

            >

              {

                loading

                  ? "Signing In..."

                  : "Login"

              }

            </button>


          </form>


          {/* ============================================== */}
          {/* REGISTER */}
          {/* ============================================== */}

          <p className="register-text">

            Don't have an account?


            <Link to="/register">

              {" "}Create Account

            </Link>

          </p>


        </div>


      </section>

    </MainLayout>

  );

}


export default Login;