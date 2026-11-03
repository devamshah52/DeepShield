import { useEffect, useMemo, useState } from "react";

import MainLayout from "../layouts/MainLayout";

import "../styles/History.css";

import { toast } from "react-toastify";

import { getHistory } from "../api/history";

import {
  getPredictionClass,
  getStatusClass,
  formatPrediction,
  formatStatus,
} from "../utils/detection";


function History() {

  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(true);

  const [searchTerm, setSearchTerm] = useState("");

  const [predictionFilter, setPredictionFilter] =
    useState("all");

  const [sortOrder, setSortOrder] =
    useState("newest");


  // ============================================================
  // Load history
  // ============================================================

  useEffect(() => {

    const fetchHistory = async () => {

      try {

        const response =
          await getHistory();

        setHistory(
          response.data
        );

      } catch (error) {

        console.error(
          "History error:",
          error
        );

        toast.error(
          "Failed to load history."
        );

      } finally {

        setLoading(false);

      }

    };


    fetchHistory();

  }, []);


  // ============================================================
  // Format uploaded time
  // ============================================================

  const formatUploadedTime = (timestamp) => {

    if (!timestamp) {
      return "--";
    }

    return new Date(
      timestamp
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
    );

  };


  // ============================================================
  // Filter + search + sort
  // ============================================================

  const filteredHistory = useMemo(() => {

    let result = [...history];


    // ----------------------------------------------------------
    // Search
    // ----------------------------------------------------------

    const search =
      searchTerm
        .trim()
        .toLowerCase();

    if (search) {

      result = result.filter(
        (item) =>
          (
            item.original_filename ||
            ""
          )
            .toLowerCase()
            .includes(search)
      );

    }


    // ----------------------------------------------------------
    // Prediction filter
    // ----------------------------------------------------------

    if (
      predictionFilter !== "all"
    ) {

      result = result.filter(
        (item) =>
          (
            item.prediction ||
            ""
          ).toLowerCase() ===
          predictionFilter.toLowerCase()
      );

    }


    // ----------------------------------------------------------
    // Sort
    // ----------------------------------------------------------

    result.sort(
      (a, b) => {

        const dateA =
          new Date(
            a.uploaded_at || 0
          ).getTime();

        const dateB =
          new Date(
            b.uploaded_at || 0
          ).getTime();

        return sortOrder === "newest"
          ? dateB - dateA
          : dateA - dateB;

      }
    );


    return result;

  }, [
    history,
    searchTerm,
    predictionFilter,
    sortOrder,
  ]);


  // ============================================================
  // Clear filters
  // ============================================================

  const clearFilters = () => {

    setSearchTerm("");

    setPredictionFilter("all");

    setSortOrder("newest");

  };


  // ============================================================
  // Render
  // ============================================================

  return (

    <MainLayout>

      <section className="history-page">


        {/* ======================================================
            PAGE HEADER
        ====================================================== */}

        <div className="history-header">

          <div>

            <span className="history-eyebrow">
              DEEPSHIELD AI
            </span>

            <h1>
              Detection History
            </h1>

            <p>
              View and manage your previous
              DeepShield audio analyses.
            </p>

          </div>


          {!loading &&
            history.length > 0 && (

              <div className="history-count">

                <strong>
                  {history.length}
                </strong>

                <span>

                  {history.length === 1
                    ? " Analysis"
                    : " Analyses"
                  }

                </span>

              </div>

            )}

        </div>


        {/* ======================================================
            LOADING
        ====================================================== */}

        {loading && (

          <div className="history-loading">

            <div className="loading-spinner"></div>

            <h3>
              Loading history...
            </h3>

            <p>
              Retrieving your previous analyses.
            </p>

          </div>

        )}


        {/* ======================================================
            EMPTY HISTORY
        ====================================================== */}

        {!loading &&
          history.length === 0 && (

            <div className="empty-history">

              <div className="empty-history-icon">
                🎵
              </div>

              <h3>
                No uploads yet
              </h3>

              <p>
                Upload your first audio file
                to begin detection.
              </p>

            </div>

          )}


        {/* ======================================================
            HISTORY CONTENT
        ====================================================== */}

        {!loading &&
          history.length > 0 && (

            <>

              {/* ==================================================
                  FILTER BAR
              ================================================== */}

              <div className="history-controls">


                {/* SEARCH */}

                <div className="history-search">

                  <span className="search-icon">
                    🔍
                  </span>

                  <input
                    type="text"
                    placeholder="Search by filename..."
                    value={searchTerm}
                    onChange={(event) =>
                      setSearchTerm(
                        event.target.value
                      )
                    }
                  />

                  {searchTerm && (

                    <button
                      type="button"
                      className="clear-search-btn"
                      onClick={() =>
                        setSearchTerm("")
                      }
                      aria-label="Clear search"
                    >
                      ×
                    </button>

                  )}

                </div>


                {/* PREDICTION FILTER */}

                <div className="history-filter">

                  <label htmlFor="prediction-filter">
                    Prediction
                  </label>

                  <select
                    id="prediction-filter"
                    value={predictionFilter}
                    onChange={(event) =>
                      setPredictionFilter(
                        event.target.value
                      )
                    }
                  >

                    <option value="all">
                      All Predictions
                    </option>

                    <option value="Genuine">
                      Genuine
                    </option>

                    <option value="TTS">
                      TTS
                    </option>

                    <option value="Voice Conversion">
                      Voice Conversion
                    </option>

                  </select>

                </div>


                {/* SORT */}

                <div className="history-filter">

                  <label htmlFor="history-sort">
                    Sort
                  </label>

                  <select
                    id="history-sort"
                    value={sortOrder}
                    onChange={(event) =>
                      setSortOrder(
                        event.target.value
                      )
                    }
                  >

                    <option value="newest">
                      Newest First
                    </option>

                    <option value="oldest">
                      Oldest First
                    </option>

                  </select>

                </div>

              </div>


              {/* ==================================================
                  RESULTS SUMMARY
              ================================================== */}

              <div className="history-results-bar">

                <span>

                  Showing{" "}

                  <strong>
                    {filteredHistory.length}
                  </strong>

                  {" "}of{" "}

                  <strong>
                    {history.length}
                  </strong>

                  {" "}analyses

                </span>


                {(searchTerm ||
                  predictionFilter !== "all" ||
                  sortOrder !== "newest") && (

                  <button
                    type="button"
                    className="clear-filters-btn"
                    onClick={clearFilters}
                  >
                    Clear Filters
                  </button>

                )}

              </div>


              {/* ==================================================
                  NO FILTER MATCH
              ================================================== */}

              {filteredHistory.length === 0 && (

                <div className="no-history-results">

                  <div>
                    🔎
                  </div>

                  <h3>
                    No matching analyses
                  </h3>

                  <p>
                    Try changing your search
                    or prediction filter.
                  </p>

                  <button
                    type="button"
                    onClick={clearFilters}
                  >
                    Clear Filters
                  </button>

                </div>

              )}


              {/* ==================================================
                  HISTORY TABLE
              ================================================== */}

              {filteredHistory.length > 0 && (

                <div className="history-table-wrapper">

                  <table className="history-table">

                    <thead>

                      <tr>

                        <th>
                          Filename
                        </th>

                        <th>
                          Status
                        </th>

                        <th>
                          Prediction
                        </th>

                        <th>
                          Confidence
                        </th>

                        <th>
                          Uploaded
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {filteredHistory.map(
                        (item) => (

                          <tr
                            key={item.id}
                          >


                            {/* ====================================
                                FILENAME
                            ==================================== */}

                            <td>

                              <div className="filename-cell">

                                <span className="file-icon">
                                  🎵
                                </span>

                                <span
                                  className="filename"
                                  title={
                                    item.original_filename
                                  }
                                >

                                  {
                                    item.original_filename
                                  }

                                </span>

                              </div>

                            </td>


                            {/* ====================================
                                STATUS
                            ==================================== */}

                            <td>

                              <span
                                className={
                                  getStatusClass(
                                    item.status
                                  )
                                }
                              >

                                {formatStatus(
                                  item.status
                                )}

                              </span>

                            </td>


                            {/* ====================================
                                PREDICTION
                            ==================================== */}

                            <td>

                              <span
                                className={
                                  getPredictionClass(
                                    item.prediction
                                  )
                                }
                              >

                                {formatPrediction(
                                  item.prediction
                                )}

                              </span>

                            </td>


                            {/* ====================================
                                CONFIDENCE
                            ==================================== */}

                            <td>

                              {item.confidence !== null &&
                              item.confidence !== undefined ? (

                                <div className="history-confidence">

                                  <strong>
                                    {Number(
                                      item.confidence
                                    ).toFixed(2)}
                                    %
                                  </strong>

                                  <div className="history-confidence-track">

                                    <div
                                      className="history-confidence-fill"
                                      style={{
                                        width: `${Math.min(
                                          Math.max(
                                            Number(
                                              item.confidence
                                            ),
                                            0
                                          ),
                                          100
                                        )}%`,
                                      }}
                                    />

                                  </div>

                                </div>

                              ) : (

                                "--"

                              )}

                            </td>


                            {/* ====================================
                                UPLOADED TIME
                            ==================================== */}

                            <td>

                              <span className="uploaded-time">

                                {formatUploadedTime(
                                  item.uploaded_at
                                )}

                              </span>

                            </td>


                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

              )}

            </>

          )}


      </section>

    </MainLayout>

  );

}


export default History;