import API from "./auth";


// ============================================================
// ADMIN LOGIN
// ============================================================

export const adminLogin = (adminData) => {

  const formData = new URLSearchParams();


  formData.append(
    "username",
    adminData.email
  );


  formData.append(
    "password",
    adminData.password
  );


  return API.post(
    "/admin/login",
    formData,
    {
      headers: {
        "Content-Type":
          "application/x-www-form-urlencoded",
      },
    }
  );

};


// ============================================================
// ADMIN DASHBOARD
// ============================================================

export const getAdminDashboard = () => {

  const token =
    localStorage.getItem("admin_token");


  return API.get(
    "/admin/dashboard",
    {
      headers: {
        Authorization:
          `Bearer ${token}`,
      },
    }
  );

};