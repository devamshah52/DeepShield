import { Link, NavLink, useNavigate } from "react-router-dom";
import { useState, useRef, useEffect } from "react";
import { toast } from "react-toastify";

import {
  FaUser,
  FaSignOutAlt,
} from "react-icons/fa";

import { useUser } from "../context/UserContext";

import "../styles/Navbar.css";


function Navbar() {

  const [menuOpen, setMenuOpen] = useState(false);

  const [dropdownOpen, setDropdownOpen] = useState(false);

  const dropdownRef = useRef(null);

  const navigate = useNavigate();

  const {
    user,
    setUser,
    isAuthenticated,
  } = useUser();


  // ============================================================
  // CLOSE DROPDOWN WHEN CLICKING OUTSIDE
  // ============================================================

  useEffect(() => {

    const handleClickOutside = (event) => {

      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target)
      ) {

        setDropdownOpen(false);

      }

    };


    document.addEventListener(
      "mousedown",
      handleClickOutside
    );


    return () => {

      document.removeEventListener(
        "mousedown",
        handleClickOutside
      );

    };

  }, []);


  // ============================================================
  // LOGOUT
  // ============================================================

  const handleLogout = () => {

    localStorage.removeItem("token");

    setUser(null);

    setDropdownOpen(false);

    setMenuOpen(false);

    toast.success("Logged out successfully!");

    setTimeout(() => {

      navigate("/login");

    }, 800);

  };


  // ============================================================
  // USER INITIALS
  // ============================================================

  const getInitials = (name) => {

    if (!name) return "";

    return name
      .split(" ")
      .map((word) => word[0])
      .join("")
      .toUpperCase();

  };


  // ============================================================
  // NAVBAR
  // ============================================================

  return (

    <nav className="navbar">


      {/* ======================================================
          LOGO
      ====================================================== */}

      <Link
        to={isAuthenticated ? "/dashboard" : "/"}
        className="logo"
      >

        DeepShield

      </Link>


      {/* ======================================================
          MOBILE MENU
      ====================================================== */}

      <button
        className="menu-toggle"
        onClick={() =>
          setMenuOpen(!menuOpen)
        }
      >

        ☰

      </button>


      {/* ======================================================
          NAVIGATION
      ====================================================== */}

      <ul
        className={`nav-links ${
          menuOpen ? "active" : ""
        }`}
      >


        {!isAuthenticated ? (

          /* ==================================================
             PUBLIC NAVIGATION
          ================================================== */

          <>

            <li>

              <NavLink
                to="/"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                Home

              </NavLink>

            </li>


            <li>

              <NavLink
                to="/about"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                About

              </NavLink>

            </li>


            <li>

              <NavLink
                to="/contact"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                Contact

              </NavLink>

            </li>

          </>

        ) : (

          /* ==================================================
             LOGGED-IN USER NAVIGATION
          ================================================== */

          <>

            <li>

              <NavLink
                to="/dashboard"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                Dashboard

              </NavLink>

            </li>


            <li>

              <NavLink
                to="/upload"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                Upload

              </NavLink>

            </li>


            <li>

              <NavLink
                to="/history"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                History

              </NavLink>

            </li>


            <li>

              <NavLink
                to="/analytics"
                className={({ isActive }) =>
                  isActive ? "active-link" : ""
                }
                onClick={() =>
                  setMenuOpen(false)
                }
              >

                Analytics

              </NavLink>

            </li>

          </>

        )}

      </ul>


      {/* ======================================================
          RIGHT SIDE
      ====================================================== */}

      <div className="auth-buttons">


        {!isAuthenticated ? (

          /* ==================================================
             LOGGED OUT
          ================================================== */

          <>

            <Link
              to="/login"
              className="login-btn"
            >

              Login

            </Link>


            <Link
              to="/register"
              className="register-btn"
            >

              Get Started

            </Link>

          </>

        ) : (

          /* ==================================================
             LOGGED IN
          ================================================== */

          <div
            className="user-menu"
            ref={dropdownRef}
          >


            {/* USER AVATAR */}

            <div
              className="user-avatar"
              title={user?.full_name}
              onClick={() =>
                setDropdownOpen(!dropdownOpen)
              }
            >

              {getInitials(user?.full_name)}

            </div>


            {/* PROFILE DROPDOWN */}

            {dropdownOpen && (

              <div className="profile-dropdown">


                <div className="dropdown-header">

                  <div className="dropdown-avatar">

                    {getInitials(
                      user?.full_name
                    )}

                  </div>


                  <h4>

                    {user?.full_name}

                  </h4>


                  <p>

                    {user?.email}

                  </p>

                </div>


                {/* DASHBOARD */}

                <button
                  className="dropdown-item"
                  onClick={() => {

                    setDropdownOpen(false);

                    navigate("/dashboard");

                  }}
                >

                  🏠

                  Dashboard

                </button>


                {/* PROFILE */}

                <button
                  className="dropdown-item"
                  onClick={() => {

                    setDropdownOpen(false);

                    navigate("/profile");

                  }}
                >

                  <FaUser />

                  My Profile

                </button>


                {/* LOGOUT */}

                <button
                  className="dropdown-item logout-item"
                  onClick={handleLogout}
                >

                  <FaSignOutAlt />

                  Logout

                </button>


              </div>

            )}

          </div>

        )}

      </div>


    </nav>

  );

}


export default Navbar;