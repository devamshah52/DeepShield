import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  FaUser,
  FaEnvelope,
  FaShieldAlt,
  FaArrowLeft,
  FaUserShield,
} from "react-icons/fa";

import { toast } from "react-toastify";

import AdminLayout from "../layouts/AdminLayout";
import { getAdminDashboard } from "../api/admin";

import "../styles/AdminProfile.css";


function AdminProfile() {

  const navigate = useNavigate();

  const [admin, setAdmin] = useState(null);
  const [loading, setLoading] = useState(true);


  // ============================================================
  // LOAD ADMIN PROFILE
  // ============================================================

  useEffect(() => {

    const loadProfile = async () => {

      try {

        const response =
          await getAdminDashboard();

        setAdmin(
          response.data?.admin || null
        );

      } catch (error) {

        console.error(
          "Admin profile error:",
          error
        );


        if (
          error.response?.status === 401 ||
          error.response?.status === 403
        ) {

          localStorage.removeItem(
            "admin_token"
          );

          toast.error(
            "Admin session expired. Please login again."
          );

          navigate("/admin/login");

          return;

        }


        toast.error(
          error.response?.data?.detail ||
          "Failed to load admin profile."
        );

      } finally {

        setLoading(false);

      }

    };


    loadProfile();

  }, [navigate]);


  // ============================================================
  // LOADING
  // ============================================================

  if (loading) {

    return (

      <AdminLayout showLogout>

        <section className="admin-profile-page">

          <div className="admin-profile-loading">

            <FaUserShield />

            <h2>
              Loading Profile...
            </h2>

            <p>
              Please wait while we load
              your administrator information.
            </p>

          </div>

        </section>

      </AdminLayout>

    );

  }


  // ============================================================
  // PROFILE NOT FOUND
  // ============================================================

  if (!admin) {

    return (

      <AdminLayout showLogout>

        <section className="admin-profile-page">

          <div className="admin-profile-loading">

            <FaShieldAlt />

            <h2>
              Profile Not Available
            </h2>

            <p>
              Administrator information could
              not be loaded.
            </p>

            <button
              className="admin-profile-back-btn"
              onClick={() =>
                navigate("/admin/dashboard")
              }
            >
              <FaArrowLeft />
              Back to Dashboard
            </button>

          </div>

        </section>

      </AdminLayout>

    );

  }


  // ============================================================
  // PROFILE
  // ============================================================

  const fullName =
    admin.full_name || "Administrator";

  const email =
    admin.email || "Not available";


  return (

    <AdminLayout showLogout>

      <section className="admin-profile-page">


        {/* ==================================================
            PAGE HEADER
            ================================================== */}

        <div className="admin-profile-heading">

          <div>

            <span className="admin-profile-label">
              ADMIN ACCOUNT
            </span>

            <h1>
              My Profile
            </h1>

            <p>
              View your administrator account
              information.
            </p>

          </div>


          <button
            className="admin-profile-back-btn"
            onClick={() =>
              navigate("/admin/dashboard")
            }
          >
            <FaArrowLeft />
            Back to Dashboard
          </button>

        </div>


        {/* ==================================================
            PROFILE CARD
            ================================================== */}

        <div className="admin-profile-card">


          {/* AVATAR */}

          <div className="admin-profile-avatar">

            {fullName
              .charAt(0)
              .toUpperCase()}

          </div>


          {/* NAME */}

          <h2>
            {fullName}
          </h2>


          <span className="admin-profile-role">

            <FaUserShield />

            Administrator

          </span>


          {/* DETAILS */}

          <div className="admin-profile-details">


            {/* FULL NAME */}

            <div className="admin-profile-detail">

              <div className="admin-profile-detail-icon">

                <FaUser />

              </div>


              <div>

                <span>
                  Full Name
                </span>

                <strong>
                  {fullName}
                </strong>

              </div>

            </div>


            {/* EMAIL */}

            <div className="admin-profile-detail">

              <div className="admin-profile-detail-icon">

                <FaEnvelope />

              </div>


              <div>

                <span>
                  Email Address
                </span>

                <strong>
                  {email}
                </strong>

              </div>

            </div>


            {/* ACCOUNT TYPE */}

            <div className="admin-profile-detail">

              <div className="admin-profile-detail-icon">

                <FaShieldAlt />

              </div>


              <div>

                <span>
                  Account Type
                </span>

                <strong>
                  Administrator
                </strong>

              </div>

            </div>


          </div>


        </div>


      </section>

    </AdminLayout>

  );

}


export default AdminProfile;