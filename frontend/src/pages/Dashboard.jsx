import { useNavigate } from "react-router-dom";

import { useUser } from "../context/UserContext";

import MainLayout from "../layouts/MainLayout";

import "../styles/Dashboard.css";

import {
  FaUpload,
  FaHistory,
  FaChartBar,
  FaFileAudio,
  FaCheckCircle,
  FaRobot,
  FaMicrophone,
} from "react-icons/fa";

import {
  getPredictionClass,
  getStatusClass,
  formatPrediction,
  formatStatus,
} from "../utils/detection";


function Dashboard() {

  const navigate = useNavigate();

  const { user, loading } = useUser();


  // ============================================================
  // Loading
  // ============================================================

  if (loading) {

    return (

      <MainLayout>

        <section className="dashboard">

          <div className="dashboard-box">

            <h2>
              Loading Dashboard...
            </h2>

          </div>

        </section>

      </MainLayout>

    );

  }


  // ============================================================
  // Dashboard
  // ============================================================

  return (

    <MainLayout>

      <section className="dashboard">


        {/* ======================================================
            WELCOME BANNER
        ====================================================== */}

        <div className="dashboard-banner">

          <div>

            <h1>

              👋 Welcome Back,

              <span>
                {" "}
                {user?.full_name || "User"}
              </span>

            </h1>


            <p>

              Manage your AI audio detection
              activities from one secure dashboard.

            </p>

          </div>


          <div className="banner-circle">

            <span>

              {user?.statistics?.total_uploads ?? 0}

            </span>

            <small>
              Uploads
            </small>

          </div>

        </div>


        {/* ======================================================
            STATISTICS
        ====================================================== */}

        <div className="dashboard-cards">


          {/* ==================================================
              TOTAL UPLOADS
          ================================================== */}

          <div className="dashboard-card">

            <FaFileAudio className="card-icon" />

            <h2>

              {user?.statistics?.total_uploads ?? 0}

            </h2>

            <p>
              Total Uploads
            </p>

          </div>


          {/* ==================================================
              GENUINE AUDIO
          ================================================== */}

          <div className="dashboard-card">

            <FaCheckCircle className="card-icon" />

            <h2>

              {user?.statistics?.genuine_audio ?? 0}

            </h2>

            <p>
              Genuine Audio
            </p>

          </div>


          {/* ==================================================
              TTS AUDIO
          ================================================== */}

          <div className="dashboard-card">

            <FaRobot className="card-icon" />

            <h2>

              {user?.statistics?.tts_audio ?? 0}

            </h2>

            <p>
              TTS Audio
            </p>

          </div>


          {/* ==================================================
              VOICE CONVERSION
          ================================================== */}

          <div className="dashboard-card">

            <FaMicrophone className="card-icon" />

            <h2>

              {
                user?.statistics
                  ?.voice_conversion_audio ?? 0
              }

            </h2>

            <p>
              Voice Conversion
            </p>

          </div>

        </div>


        {/* ======================================================
            BOTTOM SECTIONS
        ====================================================== */}

        <div className="dashboard-sections">


          {/* ==================================================
              QUICK ACTIONS
          ================================================== */}

          <div className="dashboard-box">

            <h3>
              ⚡ Quick Actions
            </h3>


            {/* Upload Audio */}

            <div
              className="quick-action-card"
              onClick={() => navigate("/upload")}
            >

              <FaUpload className="quick-icon" />

              <div>

                <h4>
                  Upload Audio
                </h4>

                <p>
                  Upload an audio file for AI analysis.
                </p>

              </div>

            </div>


            {/* Detection History */}

            <div
              className="quick-action-card"
              onClick={() => navigate("/history")}
            >

              <FaHistory className="quick-icon" />

              <div>

                <h4>
                  Detection History
                </h4>

                <p>
                  View all previous detections.
                </p>

              </div>

            </div>


            {/* Detection Analytics */}

            <div
              className="quick-action-card"
              onClick={() => navigate("/analytics")}
            >

              <FaChartBar className="quick-icon" />

              <div>

                <h4>
                  Detection Analytics
                </h4>

                <p>
                  View your detection statistics and insights.
                </p>

              </div>

            </div>

          </div>


          {/* ==================================================
              RECENT ACTIVITY
          ================================================== */}

          <div className="dashboard-box">

            <h3>
              🕒 Recent Activity
            </h3>


            {user?.recent_activity &&
            user.recent_activity.length > 0 ? (

              <div className="recent-activity-list">

                {user.recent_activity.map(
                  (activity, index) => {

                    const prediction =
                      activity.prediction;

                    const status =
                      activity.status;


                    return (

                      <div
                        className="activity-item"
                        key={
                          `${activity.title}-${activity.timestamp}-${index}`
                        }
                      >


                        {/* ==================================================
                            ACTIVITY TITLE
                        ================================================== */}

                        <h4 className="activity-title">

                          {activity.title}

                        </h4>


                        {/* ==================================================
                            PREDICTION / STATUS
                        ================================================== */}

                        <div className="activity-prediction">


                          {prediction ? (

                            <span
                              className={
                                getPredictionClass(
                                  prediction
                                )
                              }
                            >

                              {formatPrediction(
                                prediction
                              )}

                            </span>

                          ) : (

                            <span
                              className={
                                getStatusClass(
                                  status
                                )
                              }
                            >

                              {formatStatus(
                                status
                              )}

                            </span>

                          )}


                          {/* ==================================================
                              CONFIDENCE
                          ================================================== */}

                          {prediction &&
                          activity.confidence !== null &&
                          activity.confidence !== undefined && (

                            <span className="activity-confidence">

                              {Number(
                                activity.confidence
                              ).toFixed(2)}

                              % confidence

                            </span>

                          )}

                        </div>


                        {/* ==================================================
                            DESCRIPTION
                        ================================================== */}

                        {!prediction &&
                        activity.description && (

                          <p className="activity-description">

                            {activity.description}

                          </p>

                        )}


                        {/* ==================================================
                            STATUS
                        ================================================== */}

                        {prediction &&
                        status && (

                          <span
                            className={
                              getStatusClass(
                                status
                              )
                            }
                          >

                            {formatStatus(
                              status
                            )}

                          </span>

                        )}


                        {/* ==================================================
                            TIMESTAMP
                        ================================================== */}

                        <span className="activity-time">

                          {activity.timestamp

                            ? new Date(
                                activity.timestamp
                              ).toLocaleString(
                                "en-IN",
                                {
                                  timeZone:
                                    "Asia/Kolkata",

                                  day: "2-digit",

                                  month: "short",

                                  year: "numeric",

                                  hour: "2-digit",

                                  minute: "2-digit",

                                  hour12: true,
                                }
                              )

                            : "--"

                          }

                        </span>

                      </div>

                    );

                  }
                )}

              </div>

            ) : (

              <div className="empty-activity">

                <h4>
                  No activity yet
                </h4>

                <p>

                  Upload your first audio file
                  to begin detecting genuine,
                  TTS, and voice conversion audio.

                </p>

              </div>

            )}

          </div>

        </div>

      </section>

    </MainLayout>

  );

}


export default Dashboard;