import API from "./auth";


// ============================================================
// GET ALL USERS
// ============================================================

export const getAdminUsers = (search = "") => {

  const params = {};

  if (search.trim()) {

    params.search =
      search.trim();

  }

  return API.get(
    "/admin/users",
    {
      params,

      headers: {
        Authorization:
          `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }
  );

};


// ============================================================
// BLOCK USER
// ============================================================

export const blockUser = (userId) => {

  return API.patch(

    `/admin/users/${userId}/block`,

    {},

    {
      headers: {
        Authorization:
          `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }

  );

};


// ============================================================
// UNBLOCK USER
// ============================================================

export const unblockUser = (userId) => {

  return API.patch(

    `/admin/users/${userId}/unblock`,

    {},

    {
      headers: {
        Authorization:
          `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }

  );

};


// ============================================================
// DELETE USER
// ============================================================

export const deleteUser = (userId) => {

  return API.delete(

    `/admin/users/${userId}`,

    {
      headers: {
        Authorization:
          `Bearer ${localStorage.getItem("admin_token")}`,
      },
    }

  );

};