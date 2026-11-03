import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import AdminLayout from "../layouts/AdminLayout";

import { getAdminDashboard } from "../api/admin";

import { toast } from "react-toastify";

import {
  FaUsers,
  FaFileAudio,
  FaCheckCircle,
  FaTimesCircle,
  FaChartLine,
} from "react-icons/fa";

import "../styles/AdminDashboard.css";


function AdminDashboard() {

  const navigate = useNavigate();


  // ============================================================
  // STATE
  // ============================================================

  const [dashboard, setDashboard] = useState(null);

  const [loading, setLoading] = useState(true);


  // ============================================================
  // LOAD ADMIN DASHBOARD
  // ============================================================

  useEffect(() => {

    const loadDashboard = async () => {

      try {

        const response =
          await getAdminDashboard();

        setDashboard(
          response.data
        );

      } catch (error) {

        console.error(
          "Admin dashboard error:",
          error
        );


        // ------------------------------------------------------
        // ADMIN SESSION EXPIRED
        // ------------------------------------------------------

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

          navigate(
            "/admin/login"
          );

          return;

        }


        // ------------------------------------------------------
        // OTHER ERROR
        // ------------------------------------------------------

        toast.error(
          error.response?.data?.detail ||
          "Failed to load admin dashboard."
        );

      } finally {

        setLoading(false);

      }

    };


    loadDashboard();

  }, [navigate]);


  // ============================================================
  // LOADING
  // ============================================================

  if (loading) {

    return (

      <AdminLayout showLogout>

        <section className="admin-dashboard admin-theme">

          <div className="admin-loading">

            <div className="admin-loading-icon">

              <FaChartLine />

            </div>


            <h2>

              Loading Admin Dashboard...

            </h2>


            <p>

              Please wait while we load
              the administration data.

            </p>

          </div>

        </section>

      </AdminLayout>

    );

  }


  // ============================================================
  // NO DATA
  // ============================================================

  if (!dashboard) {

    return (

      <AdminLayout showLogout>

        <section className="admin-dashboard admin-theme">

          <div className="admin-loading">

            <div className="admin-loading-icon">

              <FaTimesCircle />

            </div>


            <h2>

              Unable to load dashboard

            </h2>


            <p>

              We could not load the admin
              dashboard data.

            </p>


            <button
              className="admin-back-login-btn"
              onClick={() =>
                navigate("/admin/login")
              }
            >

              Back to Admin Login

            </button>

          </div>

        </section>

      </AdminLayout>

    );

  }


  // ============================================================
  // DASHBOARD DATA
  // ============================================================

  const statistics =
    dashboard.statistics || {};

  const recentActivity =
    dashboard.recent_activity || [];


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <AdminLayout showLogout>

      <section className="admin-dashboard admin-theme">


        {/* ==================================================
            STATISTICS
            ================================================== */}

        <div className="admin-statistics">


          {/* TOTAL USERS */}

          <div className="admin-stat-card">

            <div className="admin-stat-icon">

              <FaUsers />

            </div>


            <div className="admin-stat-content">

              <h2>

                {statistics.total_users ?? 0}

              </h2>


              <p>

                Total Users

              </p>

            </div>

          </div>


          {/* TOTAL UPLOADS */}

          <div className="admin-stat-card">

            <div className="admin-stat-icon">

              <FaFileAudio />

            </div>


            <div className="admin-stat-content">

              <h2>

                {statistics.total_uploads ?? 0}

              </h2>


              <p>

                Total Uploads

              </p>

            </div>

          </div>


          {/* COMPLETED */}

          <div className="admin-stat-card">

            <div className="admin-stat-icon">

              <FaCheckCircle />

            </div>


            <div className="admin-stat-content">

              <h2>

                {statistics.completed_uploads ?? 0}

              </h2>


              <p>

                Completed

              </p>

            </div>

          </div>


          {/* FAILED */}

          <div className="admin-stat-card">

            <div className="admin-stat-icon">

              <FaTimesCircle />

            </div>


            <div className="admin-stat-content">

              <h2>

                {statistics.failed_uploads ?? 0}

              </h2>


              <p>

                Failed

              </p>

            </div>

          </div>


        </div>


        {/* ==================================================
            RECENT ACTIVITY
            ================================================== */}

        <div className="admin-activity-box">


          <div className="admin-section-title">

            <div>

              <h2>

                Recent Activity

              </h2>


              <p>

                Latest audio detection activity
                across the platform.

              </p>

            </div>


            <FaFileAudio />

          </div>


          {/* EMPTY ACTIVITY */}

          {recentActivity.length === 0 ? (

            <div className="admin-empty-activity">

              <div className="admin-empty-icon">

                <FaFileAudio />

              </div>


              <h3>

                No Recent Activity

              </h3>


              <p>

                No audio detections have been
                recorded yet.

              </p>

            </div>

          ) : (

            /* ==================================================
               ACTIVITY TABLE
               ================================================== */

            <div className="admin-activity-table-wrapper">

              <table className="admin-activity-table">

                <thead>

                  <tr>

                    <th>
                      File
                    </th>


                    <th>
                      User
                    </th>


                    <th>
                      Prediction
                    </th>


                    <th>
                      Confidence
                    </th>


                    <th>
                      Status
                    </th>


                    <th>
                      Uploaded At
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {recentActivity.map(
                    (activity) => (

                      <tr
                        key={activity.id}
                      >


                        {/* FILE */}

                        <td>

                          <span className="admin-file-name">

                            {activity.filename}

                          </span>

                        </td>


                        {/* USER */}

                        <td>

                          <span className="admin-user-name">

                            {activity.user_name ||
                              "Unknown User"}

                          </span>

                        </td>


                        {/* PREDICTION */}

                        <td>

                          <span className="prediction-badge">

                            {activity.prediction ||
                              "Pending"}

                          </span>

                        </td>


                        {/* CONFIDENCE */}

                        <td>

                          <span className="confidence-value">

                            {activity.confidence != null
                              ? `${activity.confidence}%`
                              : "N/A"}

                          </span>

                        </td>


                        {/* STATUS */}

                        <td>

                          <span
                            className={
                              activity.status ===
                              "completed"
                                ? "status-badge status-completed"
                                : "status-badge"
                            }
                          >

                            {activity.status}

                          </span>

                        </td>


                        {/* UPLOADED AT */}

                        <td>

                          <span className="admin-date">

                            {activity.uploaded_at
                              ? new Date(
                                  activity.uploaded_at
                                ).toLocaleString()
                              : "N/A"}

                          </span>

                        </td>


                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          )}

        </div>


      </section>

    </AdminLayout>

  );

}


export default AdminDashboard;