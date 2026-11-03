import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";
import { getAnalytics } from "../api/analytics";

import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
} from "recharts";

import {
  FaArrowLeft,
  FaChartPie,
  FaChartLine,
  FaChartBar,
  FaFileAudio,
  FaCheckCircle,
  FaRobot,
  FaMicrophone,
  FaBullseye,
} from "react-icons/fa";

import "../styles/Analytics.css";


function Analytics() {

  const navigate = useNavigate();

  const [analytics, setAnalytics] = useState(null);

  const [loading, setLoading] = useState(true);

  const [error, setError] = useState("");


  // ============================================================
  // LOAD ANALYTICS
  // ============================================================

  useEffect(() => {

    const loadAnalytics = async () => {

      try {

        setLoading(true);

        setError("");

        const response = await getAnalytics();

        setAnalytics(
          response.data
        );

      } catch (err) {

        console.error(
          "Failed to load analytics:",
          err
        );

        setError(
          err?.response?.data?.detail ||
          "Failed to load detection analytics."
        );

      } finally {

        setLoading(false);

      }

    };


    loadAnalytics();

  }, []);


  // ============================================================
  // LOADING
  // ============================================================

  if (loading) {

    return (

      <MainLayout>

        <section className="analytics-page">

          <div className="analytics-loading">

            <FaChartBar />

            <h2>
              Loading Detection Analytics...
            </h2>

            <p>
              Analyzing your detection history.
            </p>

          </div>

        </section>

      </MainLayout>

    );

  }


  // ============================================================
  // ERROR
  // ============================================================

  if (error || !analytics) {

    return (

      <MainLayout>

        <section className="analytics-page">

          <div className="analytics-error">

            <h2>
              Unable to Load Analytics
            </h2>

            <p>
              {error || "No analytics data is available."}
            </p>

            <button
              className="analytics-back-button"
              onClick={() => navigate("/dashboard")}
            >
              <FaArrowLeft />
              Back to Dashboard
            </button>

          </div>

        </section>

      </MainLayout>

    );

  }


  // ============================================================
  // DATA
  // ============================================================

  const summary =
    analytics.summary || {};

  const predictionDistribution =
    analytics.prediction_distribution || [];

  const confidenceByPrediction =
    analytics.average_confidence_by_prediction || [];

  const detectionTimeline =
    analytics.detection_timeline || [];

  const formatDistribution =
    analytics.format_distribution || [];


  // ============================================================
  // PREDICTION COLORS
  // ============================================================

  const predictionColors = {

    Genuine: "#22c55e",

    TTS: "#f59e0b",

    "Voice Conversion": "#ef4444",

  };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <MainLayout>

      <section className="analytics-page">


        {/* ======================================================
            HEADER
        ====================================================== */}

        <div className="analytics-header">

          <div>

            <h1>
              Detection Analysis
            </h1>

            <p>
              Analyze your personal audio detection history
              and AI prediction results.
            </p>

          </div>


          <button
            className="analytics-back-button"
            onClick={() => navigate("/dashboard")}
          >

            <FaArrowLeft />

            Back to Dashboard

          </button>

        </div>


        {/* ======================================================
            SUMMARY CARDS
        ====================================================== */}

        <div className="analytics-summary-grid">


          <div className="analytics-summary-card">

            <div className="analytics-card-icon">

              <FaFileAudio />

            </div>

            <div>

              <h2>
                {summary.total_uploads ?? 0}
              </h2>

              <p>
                Total Detections
              </p>

            </div>

          </div>


          <div className="analytics-summary-card">

            <div className="analytics-card-icon">

              <FaBullseye />

            </div>

            <div>

              <h2>
                {Number(
                  summary.average_confidence ?? 0
                ).toFixed(2)}
                %
              </h2>

              <p>
                Average Confidence
              </p>

            </div>

          </div>


          <div className="analytics-summary-card">

            <div className="analytics-card-icon">

              <FaCheckCircle />

            </div>

            <div>

              <h2>
                {summary.completed ?? 0}
              </h2>

              <p>
                Completed
              </p>

            </div>

          </div>


          <div className="analytics-summary-card">

            <div className="analytics-card-icon">

              <FaChartBar />

            </div>

            <div>

              <h2>
                {summary.failed ?? 0}
              </h2>

              <p>
                Failed
              </p>

            </div>

          </div>

        </div>


        {/* ======================================================
            DETECTION TYPE + TIMELINE
        ====================================================== */}

        <div className="analytics-grid">


          {/* ==================================================
              DETECTION TYPE
          ================================================== */}

          <div className="analytics-box">

            <div className="analytics-box-header">

              <div>

                <FaChartPie />

                <h3>
                  Detection Type Distribution
                </h3>

              </div>

            </div>


            <div className="analytics-chart">

              {predictionDistribution.length > 0 ? (

                <ResponsiveContainer
                  width="100%"
                  height={320}
                >

                  <PieChart>

                    <Pie
                      data={predictionDistribution}
                      dataKey="count"
                      nameKey="prediction"
                      cx="50%"
                      cy="50%"
                      outerRadius={105}
                      innerRadius={55}
                      paddingAngle={3}
                    >

                      {predictionDistribution.map(
                        (entry) => (

                          <Cell
                            key={entry.prediction}
                            fill={
                              predictionColors[
                                entry.prediction
                              ] || "#64748b"
                            }
                          />

                        )
                      )}

                    </Pie>

                    <Tooltip />

                    <Legend />

                  </PieChart>

                </ResponsiveContainer>

              ) : (

                <div className="analytics-empty">

                  No detection data available.

                </div>

              )}

            </div>

          </div>


          {/* ==================================================
              DETECTION TIMELINE
          ================================================== */}

          <div className="analytics-box">

            <div className="analytics-box-header">

              <div>

                <FaChartLine />

                <h3>
                  Detection Activity Over Time
                </h3>

              </div>

            </div>


            <div className="analytics-chart">

              {detectionTimeline.length > 0 ? (

                <ResponsiveContainer
                  width="100%"
                  height={320}
                >

                  <LineChart
                    data={detectionTimeline}
                    margin={{
                      top: 10,
                      right: 20,
                      left: 0,
                      bottom: 10,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      opacity={0.15}
                    />

                    <XAxis
                      dataKey="date"
                    />

                    <YAxis
                      allowDecimals={false}
                    />

                    <Tooltip />

                    <Line
                      type="monotone"
                      dataKey="count"
                      name="Detections"
                      stroke="#38bdf8"
                      strokeWidth={3}
                      dot={{
                        r: 4,
                      }}
                      activeDot={{
                        r: 6,
                      }}
                    />

                  </LineChart>

                </ResponsiveContainer>

              ) : (

                <div className="analytics-empty">

                  No timeline data available.

                </div>

              )}

            </div>

          </div>

        </div>


        {/* ======================================================
            CONFIDENCE + FORMAT
        ====================================================== */}

        <div className="analytics-grid">


          {/* ==================================================
              CONFIDENCE BY CLASS
          ================================================== */}

          <div className="analytics-box">

            <div className="analytics-box-header">

              <div>

                <FaBullseye />

                <h3>
                  Average Confidence by Detection Type
                </h3>

              </div>

            </div>


            <div className="analytics-chart">

              {confidenceByPrediction.length > 0 ? (

                <ResponsiveContainer
                  width="100%"
                  height={320}
                >

                  <BarChart
                    data={confidenceByPrediction}
                    margin={{
                      top: 10,
                      right: 20,
                      left: 0,
                      bottom: 10,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      opacity={0.15}
                    />

                    <XAxis
                      dataKey="prediction"
                    />

                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={
                        (value) =>
                          `${value}%`
                      }
                    />

                    <Tooltip
                      formatter={
                        (value) =>
                          `${Number(value).toFixed(2)}%`
                      }
                    />

                    <Bar
                      dataKey="confidence"
                      name="Confidence"
                      radius={[
                        8,
                        8,
                        0,
                        0,
                      ]}
                    >

                      {confidenceByPrediction.map(
                        (entry) => (

                          <Cell
                            key={entry.prediction}
                            fill={
                              predictionColors[
                                entry.prediction
                              ] || "#64748b"
                            }
                          />

                        )
                      )}

                    </Bar>

                  </BarChart>

                </ResponsiveContainer>

              ) : (

                <div className="analytics-empty">

                  No confidence data available.

                </div>

              )}

            </div>

          </div>


          {/* ==================================================
              AUDIO FORMAT
          ================================================== */}

          <div className="analytics-box">

            <div className="analytics-box-header">

              <div>

                <FaFileAudio />

                <h3>
                  Audio Format Distribution
                </h3>

              </div>

            </div>


            <div className="analytics-chart">

              {formatDistribution.length > 0 ? (

                <ResponsiveContainer
                  width="100%"
                  height={320}
                >

                  <BarChart
                    data={formatDistribution}
                    margin={{
                      top: 10,
                      right: 20,
                      left: 0,
                      bottom: 10,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      opacity={0.15}
                    />

                    <XAxis
                      dataKey="format"
                    />

                    <YAxis
                      allowDecimals={false}
                    />

                    <Tooltip />

                    <Bar
                      dataKey="count"
                      name="Uploads"
                      fill="#a78bfa"
                      radius={[
                        8,
                        8,
                        0,
                        0,
                      ]}
                    />

                  </BarChart>

                </ResponsiveContainer>

              ) : (

                <div className="analytics-empty">

                  No audio format data available.

                </div>

              )}

            </div>

          </div>

        </div>


        {/* ======================================================
            DETECTION BREAKDOWN
        ====================================================== */}

        <div className="analytics-box analytics-breakdown-box">

          <div className="analytics-box-header">

            <div>

              <FaChartBar />

              <h3>
                Detection Breakdown
              </h3>

            </div>

          </div>


          <div className="analytics-breakdown-grid">


            <div className="analytics-breakdown-item genuine">

              <FaCheckCircle />

              <div>

                <strong>
                  Genuine
                </strong>

                <span>
                  {summary.total_uploads
                    ? (
                        (
                          (predictionDistribution.find(
                            (item) =>
                              item.prediction ===
                              "Genuine"
                          )?.count || 0)
                          /
                          summary.total_uploads
                        ) * 100
                      ).toFixed(1)
                    : "0.0"
                  }%
                </span>

              </div>

            </div>


            <div className="analytics-breakdown-item tts">

              <FaRobot />

              <div>

                <strong>
                  TTS
                </strong>

                <span>
                  {summary.total_uploads
                    ? (
                        (
                          (predictionDistribution.find(
                            (item) =>
                              item.prediction ===
                              "TTS"
                          )?.count || 0)
                          /
                          summary.total_uploads
                        ) * 100
                      ).toFixed(1)
                    : "0.0"
                  }%
                </span>

              </div>

            </div>


            <div className="analytics-breakdown-item voice">

              <FaMicrophone />

              <div>

                <strong>
                  Voice Conversion
                </strong>

                <span>
                  {summary.total_uploads
                    ? (
                        (
                          (predictionDistribution.find(
                            (item) =>
                              item.prediction ===
                              "Voice Conversion"
                          )?.count || 0)
                          /
                          summary.total_uploads
                        ) * 100
                      ).toFixed(1)
                    : "0.0"
                  }%
                </span>

              </div>

            </div>


          </div>

        </div>


      </section>

    </MainLayout>

  );

}


export default Analytics;