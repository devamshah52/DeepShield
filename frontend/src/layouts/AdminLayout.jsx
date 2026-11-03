import { useNavigate } from "react-router-dom";

import {
  FaSignOutAlt,
  FaUserCog,
  FaUser,
} from "react-icons/fa";

import "../styles/AdminLayout.css";


function AdminLayout({
  children,
  showLogout = false,
}) {

  const navigate = useNavigate();


  // ============================================================
  // ADMIN LOGOUT
  // ============================================================

  const handleLogout = () => {

    localStorage.removeItem(
      "admin_token"
    );

    navigate(
      "/admin/login"
    );

  };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <div className="admin-layout">


      {/* ======================================================
          ADMIN HEADER
      ====================================================== */}

      {showLogout && (

        <header className="admin-header">


          {/* ==================================================
              ADMIN LOGO
          ================================================== */}

          <div className="admin-header-logo">

            <span className="admin-logo-name">
              DeepShield
            </span>

            <span className="admin-logo-badge">
              Admin
            </span>

          </div>


          {/* ==================================================
              ADMIN ACTIONS
          ================================================== */}

          <div className="admin-header-actions">


            {/* MANAGE USERS */}

            <button
              type="button"
              className="admin-header-action admin-manage-btn"
              onClick={() =>
                navigate("/admin/users")
              }
            >

              <FaUserCog />

              <span>
                Manage Users
              </span>

            </button>


            {/* PROFILE */}

            <button
              type="button"
              className="admin-header-action admin-profile-btn"
              onClick={() =>
                navigate("/admin/profile")
              }
            >

              <FaUser />

              <span>
                Profile
              </span>

            </button>


            {/* LOGOUT */}

            <button
              type="button"
              className="admin-header-logout"
              onClick={handleLogout}
            >

              <FaSignOutAlt />

              <span>
                Logout
              </span>

            </button>


          </div>


        </header>

      )}


      {/* ======================================================
          ADMIN PAGE CONTENT
      ====================================================== */}

      <main className="admin-layout-content">

        {children}

      </main>


    </div>

  );

}


export default AdminLayout;