import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

import AdminLayout from "../layouts/AdminLayout";

import "../styles/AdminLogin.css";

import { adminLogin } from "../api/admin";

import { toast } from "react-toastify";


function AdminLogin() {

  const navigate = useNavigate();


  // ============================================================
  // STATE
  // ============================================================

  const [email, setEmail] = useState("");

  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);


  const [errors, setErrors] = useState({
    email: "",
    password: "",
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


    if (!emailRegex.test(value.trim())) {

      return "Please enter a valid email address.";

    }


    return "";

  };


  // ============================================================
  // PASSWORD VALIDATION
  // ============================================================

  const validatePassword = (value) => {

    if (!value.trim()) {

      return "Password is required.";

    }


    return "";

  };


  // ============================================================
  // HANDLE LOGIN
  // ============================================================

  const handleLogin = async (e) => {

    e.preventDefault();


    // ----------------------------------------------------------
    // Validate form
    // ----------------------------------------------------------

    const newErrors = {

      email: validateEmail(email),

      password: validatePassword(password),

    };


    setErrors(newErrors);


    if (
      newErrors.email ||
      newErrors.password
    ) {

      return;

    }


    // ----------------------------------------------------------
    // Login
    // ----------------------------------------------------------

    try {

      setLoading(true);


      const response = await adminLogin({

        email:
          email.trim().toLowerCase(),

        password,

      });


      // --------------------------------------------------------
      // Get admin access token
      // --------------------------------------------------------

      const token =
        response?.data?.access_token;


      if (!token) {

        throw new Error(
          "Admin access token was not returned."
        );

      }


      // --------------------------------------------------------
      // Store ADMIN token separately
      // --------------------------------------------------------

      localStorage.setItem(
        "admin_token",
        token
      );


      // --------------------------------------------------------
      // Success message
      // --------------------------------------------------------

      toast.success(
        "Admin Login Successful!"
      );


      // --------------------------------------------------------
      // Go to Admin Dashboard
      // --------------------------------------------------------

      navigate(
        "/admin/dashboard"
      );


    } catch (error) {

      console.error(
        "Admin login error:",
        error
      );


      const errorMessage =
        error?.response?.data?.detail;


      if (errorMessage) {

        toast.error(
          errorMessage
        );

      } else if (error?.message) {

        toast.error(
          error.message
        );

      } else {

        toast.error(
          "Admin login failed."
        );

      }

    } finally {

      setLoading(false);

    }

  };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <AdminLayout>

      <section className="admin-login-page">


        {/* ================================================== */}
        {/* LEFT PANEL */}
        {/* ================================================== */}

        <div className="admin-login-left">


          <div className="admin-badge">

            ADMIN PANEL

          </div>


          <h1>

            DeepShield
            <br />
            Administration

          </h1>


          <p>

            Securely manage DeepShield users,
            audio detections, system statistics,
            and platform activity from the
            administrator dashboard.

          </p>


          <div className="admin-features">


            <div>

              <span>✓</span>

              <p>
                User Management
              </p>

            </div>


            <div>

              <span>✓</span>

              <p>
                Detection Monitoring
              </p>

            </div>


            <div>

              <span>✓</span>

              <p>
                System Analytics
              </p>

            </div>


            <div>

              <span>✓</span>

              <p>
                Secure Administration
              </p>

            </div>


          </div>

        </div>


        {/* ================================================== */}
        {/* ADMIN LOGIN CARD */}
        {/* ================================================== */}

        <div className="admin-login-container">


          <h2>

            Admin Login

          </h2>


          <p className="admin-login-subtitle">

            Sign in to access the
            DeepShield administration panel.

          </p>


          <form
            className="admin-login-form"
            onSubmit={handleLogin}
          >


            {/* ============================================ */}
            {/* EMAIL */}
            {/* ============================================ */}

            <div className="admin-form-group">

              <label htmlFor="admin-email">

                Admin Email

              </label>


              <input
                id="admin-email"
                type="email"
                placeholder="Enter admin email"
                value={email}
                autoComplete="username"
                onChange={(e) => {

                  const value =
                    e.target.value;


                  setEmail(value);


                  setErrors((previous) => ({

                    ...previous,

                    email:
                      validateEmail(value),

                  }));

                }}
                className={
                  errors.email
                    ? "admin-error-input"
                    : ""
                }
              />


              {errors.email && (

                <small className="admin-error-text">

                  {errors.email}

                </small>

              )}

            </div>


            {/* ============================================ */}
            {/* PASSWORD */}
            {/* ============================================ */}

            <div className="admin-form-group">

              <label htmlFor="admin-password">

                Password

              </label>


              <input
                id="admin-password"
                type="password"
                placeholder="Enter admin password"
                value={password}
                autoComplete="current-password"
                onChange={(e) => {

                  const value =
                    e.target.value;


                  setPassword(value);


                  setErrors((previous) => ({

                    ...previous,

                    password:
                      validatePassword(value),

                  }));

                }}
                className={
                  errors.password
                    ? "admin-error-input"
                    : ""
                }
              />


              {errors.password && (

                <small className="admin-error-text">

                  {errors.password}

                </small>

              )}

            </div>


            {/* ============================================ */}
            {/* LOGIN BUTTON */}
            {/* ============================================ */}

            <button
              type="submit"
              className="admin-login-btn"
              disabled={loading}
            >

              {loading
                ? "Signing In..."
                : "Login as Admin"
              }

            </button>


          </form>


          {/* ============================================== */}
          {/* BACK TO MAIN WEBSITE */}
          {/* ============================================== */}

          <p className="admin-back-text">

            <Link to="/">

              ← Back to DeepShield

            </Link>

          </p>


        </div>


      </section>

    </AdminLayout>

  );

}


export default AdminLogin;