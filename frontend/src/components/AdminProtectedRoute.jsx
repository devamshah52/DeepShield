import { Navigate } from "react-router-dom";


// ============================================================
// ADMIN PROTECTED ROUTE
// ============================================================

function AdminProtectedRoute({ children }) {

  const adminToken =
    localStorage.getItem("admin_token");


  // ==========================================================
  // ADMIN NOT LOGGED IN
  // ==========================================================

  if (!adminToken) {

    return (
      <Navigate
        to="/admin/login"
        replace
      />
    );

  }


  // ==========================================================
  // ADMIN LOGGED IN
  // ==========================================================

  return children;

}


export default AdminProtectedRoute;