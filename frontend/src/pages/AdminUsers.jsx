import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import AdminLayout from "../layouts/AdminLayout";

import {
  getAdminUsers,
  blockUser,
  unblockUser,
  deleteUser,
} from "../api/adminUsers";

import { toast } from "react-toastify";

import {
  FaUsers,
  FaSearch,
  FaEnvelope,
  FaUpload,
  FaUser,
  FaSyncAlt,
  FaBan,
  FaUnlock,
  FaTrash,
} from "react-icons/fa";

import "../styles/AdminUsers.css";


function AdminUsers() {

  const navigate = useNavigate();


  // ============================================================
  // STATE
  // ============================================================

  const [users, setUsers] = useState([]);

  const [search, setSearch] = useState("");

  const [loading, setLoading] = useState(true);

  const [refreshing, setRefreshing] = useState(false);

  const [actionLoading, setActionLoading] = useState(null);


  // ============================================================
  // LOAD USERS
  // ============================================================

  const loadUsers = async (searchValue = "") => {

    try {

      const response =
        await getAdminUsers(searchValue);

      const data =
        response?.data;


      if (Array.isArray(data)) {

        setUsers(data);

      } else if (Array.isArray(data?.users)) {

        setUsers(data.users);

      } else {

        setUsers([]);

      }

    } catch (error) {

      console.error(
        "Admin users error:",
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


        navigate(
          "/admin/login"
        );


        return;

      }


      toast.error(
        error.response?.data?.detail ||
        "Failed to load users."
      );

    }

  };


  // ============================================================
  // INITIAL LOAD
  // ============================================================

  useEffect(() => {

    const loadInitialUsers = async () => {

      setLoading(true);

      await loadUsers();

      setLoading(false);

    };


    loadInitialUsers();

  }, []);


  // ============================================================
  // SEARCH
  // ============================================================

  const handleSearch = async (e) => {

    const value =
      e.target.value;


    setSearch(value);


    await loadUsers(value);

  };


  // ============================================================
  // REFRESH
  // ============================================================

  const handleRefresh = async () => {

    try {

      setRefreshing(true);

      await loadUsers(search);

    } finally {

      setRefreshing(false);

    }

  };


  // ============================================================
  // BLOCK USER
  // ============================================================

  const handleBlock = async (user) => {

    const userId =
      user.id || user._id;


    if (!userId) {

      toast.error(
        "Invalid user ID."
      );

      return;

    }


    const confirmed =
      window.confirm(
        `Are you sure you want to block ${
          user.full_name || "this user"
        }?`
      );


    if (!confirmed) {

      return;

    }


    try {

      setActionLoading(
        `block-${userId}`
      );


      const response =
        await blockUser(userId);


      toast.success(
        response?.data?.message ||
        "User blocked successfully."
      );


      await loadUsers(search);

    } catch (error) {

      console.error(
        "Block user error:",
        error
      );


      if (
        error.response?.status === 401 ||
        error.response?.status === 403
      ) {

        if (
          error.response?.status === 403 &&
          error.response?.data?.detail ===
            "Admin access required."
        ) {

          localStorage.removeItem(
            "admin_token"
          );


          navigate(
            "/admin/login"
          );


          return;

        }

      }


      toast.error(
        error.response?.data?.detail ||
        "Failed to block user."
      );

    } finally {

      setActionLoading(null);

    }

  };


  // ============================================================
  // UNBLOCK USER
  // ============================================================

  const handleUnblock = async (user) => {

    const userId =
      user.id || user._id;


    if (!userId) {

      toast.error(
        "Invalid user ID."
      );

      return;

    }


    try {

      setActionLoading(
        `unblock-${userId}`
      );


      const response =
        await unblockUser(userId);


      toast.success(
        response?.data?.message ||
        "User unblocked successfully."
      );


      await loadUsers(search);

    } catch (error) {

      console.error(
        "Unblock user error:",
        error
      );


      if (
        error.response?.status === 401
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


      toast.error(
        error.response?.data?.detail ||
        "Failed to unblock user."
      );

    } finally {

      setActionLoading(null);

    }

  };


  // ============================================================
  // DELETE USER
  // ============================================================

  const handleDelete = async (user) => {

    const userId =
      user.id || user._id;


    if (!userId) {

      toast.error(
        "Invalid user ID."
      );

      return;

    }


    const confirmed =
      window.confirm(
        `Are you sure you want to permanently delete ${
          user.full_name || "this user"
        }?\n\nThis will also delete all upload records belonging to this user.`
      );


    if (!confirmed) {

      return;

    }


    try {

      setActionLoading(
        `delete-${userId}`
      );


      const response =
        await deleteUser(userId);


      toast.success(
        response?.data?.message ||
        "User deleted successfully."
      );


      await loadUsers(search);

    } catch (error) {

      console.error(
        "Delete user error:",
        error
      );


      if (
        error.response?.status === 401
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


      toast.error(
        error.response?.data?.detail ||
        "Failed to delete user."
      );


    } finally {

      setActionLoading(null);

    }

  };


  // ============================================================
  // LOADING
  // ============================================================

  if (loading) {

    return (

      <AdminLayout showLogout>

        <section className="admin-users-page">

          <div className="admin-users-loading">

            <div className="admin-users-loading-icon">

              <FaUsers />

            </div>


            <h2>

              Loading Users...

            </h2>


            <p>

              Please wait while we load
              registered users.

            </p>

          </div>

        </section>

      </AdminLayout>

    );

  }


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <AdminLayout showLogout>

      <section className="admin-users-page">


        {/* ==================================================
            PAGE HEADER
            ================================================== */}

        <div className="admin-users-header">

          <div>

            <span className="admin-users-badge">

              USER MANAGEMENT

            </span>


            <h1>

              Registered Users

            </h1>


            <p>

              View and manage users registered
              on the DeepShield platform.

            </p>

          </div>


          <div className="admin-users-count">

            <FaUsers />

            <div>

              <strong>

                {users.length}

              </strong>


              <span>

                Users

              </span>

            </div>

          </div>

        </div>


        {/* ==================================================
            SEARCH + REFRESH
            ================================================== */}

        <div className="admin-users-toolbar">


          <div className="admin-users-search">

            <FaSearch />


            <input
              type="text"
              placeholder="Search by name or email..."
              value={search}
              onChange={handleSearch}
            />

          </div>


          <button
            type="button"
            className="admin-users-refresh"
            onClick={handleRefresh}
            disabled={refreshing}
          >

            <FaSyncAlt
              className={
                refreshing
                  ? "admin-refresh-spinning"
                  : ""
              }
            />


            {refreshing
              ? "Refreshing..."
              : "Refresh"
            }

          </button>


        </div>


        {/* ==================================================
            USERS TABLE
            ================================================== */}

        <div className="admin-users-card">


          {users.length === 0 ? (

            <div className="admin-users-empty">

              <div className="admin-users-empty-icon">

                <FaUsers />

              </div>


              <h2>

                No Users Found

              </h2>


              <p>

                {search.trim()
                  ? "No users match your search."
                  : "No registered users are available yet."
                }

              </p>

            </div>

          ) : (

            <div className="admin-users-table-wrapper">

              <table className="admin-users-table">

                <thead>

                  <tr>

                    <th>
                      User
                    </th>

                    <th>
                      Email
                    </th>

                    <th>
                      Role
                    </th>

                    <th>
                      Uploads
                    </th>

                    <th>
                      Actions
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {users.map(
                    (user, index) => {

                      const userName =
                        user.full_name ||
                        user.name ||
                        "Unknown User";


                      const userEmail =
                        user.email ||
                        "N/A";


                      const userRole =
                        user.role ||
                        "user";


                      const uploadCount =
                        user.upload_count ??
                        user.total_uploads ??
                        0;


                      const userId =
                        user.id ||
                        user._id;


                      const isBlocked =
                        user.is_blocked === true;


                      const isBlockLoading =
                        actionLoading ===
                        `block-${userId}`;


                      const isUnblockLoading =
                        actionLoading ===
                        `unblock-${userId}`;


                      const isDeleteLoading =
                        actionLoading ===
                        `delete-${userId}`;


                      return (

                        <tr
                          key={
                            userId ||
                            index
                          }
                        >


                          {/* ==================================================
                              USER
                              ================================================== */}

                          <td>

                            <div className="admin-user-name">

                              <div className="admin-user-avatar">

                                {userName
                                  .charAt(0)
                                  .toUpperCase()}

                              </div>


                              <div>

                                <strong>

                                  {userName}

                                </strong>


                                <span>

                                  ID:{" "}

                                  {userId ||
                                    "N/A"}

                                </span>

                              </div>

                            </div>

                          </td>


                          {/* ==================================================
                              EMAIL
                              ================================================== */}

                          <td>

                            <div className="admin-user-email">

                              <FaEnvelope />


                              <span>

                                {userEmail}

                              </span>

                            </div>

                          </td>


                          {/* ==================================================
                              ROLE
                              ================================================== */}

                          <td>

                            <span className="admin-role-badge">

                              <FaUser />

                              {userRole}

                            </span>

                          </td>


                          {/* ==================================================
                              UPLOADS
                              ================================================== */}

                          <td>

                            <div className="admin-upload-count">

                              <FaUpload />


                              <strong>

                                {uploadCount}

                              </strong>

                            </div>

                          </td>


                          {/* ==================================================
                              ACTIONS
                              ================================================== */}

                          <td>

                            <div className="admin-user-actions">


                              {/* ==================================================
                                  BLOCK / UNBLOCK
                                  ================================================== */}

                              {isBlocked ? (

                                <button
                                  type="button"
                                  className="
                                    admin-action-btn
                                    admin-unblock-btn
                                  "
                                  onClick={() =>
                                    handleUnblock(
                                      user
                                    )
                                  }
                                  disabled={
                                    actionLoading !==
                                    null
                                  }
                                  title="Unblock User"
                                >

                                  <span className="admin-action-icon">

                                    <FaUnlock />

                                  </span>


                                  <span className="admin-action-text">

                                    {isUnblockLoading
                                      ? "Unblocking..."
                                      : "Unblock"
                                    }

                                  </span>

                                </button>

                              ) : (

                                <button
                                  type="button"
                                  className="
                                    admin-action-btn
                                    admin-block-btn
                                  "
                                  onClick={() =>
                                    handleBlock(
                                      user
                                    )
                                  }
                                  disabled={
                                    actionLoading !==
                                    null
                                  }
                                  title="Block User"
                                >

                                  <span className="admin-action-icon">

                                    <FaBan />

                                  </span>


                                  <span className="admin-action-text">

                                    {isBlockLoading
                                      ? "Blocking..."
                                      : "Block"
                                    }

                                  </span>

                                </button>

                              )}


                              {/* ==================================================
                                  DELETE
                                  ================================================== */}

                              <button
                                type="button"
                                className="
                                  admin-action-btn
                                  admin-delete-btn
                                "
                                onClick={() =>
                                  handleDelete(
                                    user
                                  )
                                }
                                disabled={
                                  actionLoading !==
                                  null
                                }
                                title="Delete User"
                              >

                                <span className="admin-action-icon">

                                  <FaTrash />

                                </span>


                                <span className="admin-action-text">

                                  {isDeleteLoading
                                    ? "Deleting..."
                                    : "Delete"
                                  }

                                </span>

                              </button>


                            </div>

                          </td>


                        </tr>

                      );

                    }
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


export default AdminUsers;