import API from "./auth";

/* ===========================
   DASHBOARD
=========================== */

export const getDashboard = () => {

  const token = localStorage.getItem("token");

  return API.get(
    "/dashboard/",
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );

};