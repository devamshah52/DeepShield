import { BrowserRouter, Routes, Route } from "react-router-dom";

// ============================
// PUBLIC PAGES
// ============================

import Home from "../pages/Home";
import About from "../pages/About";
import Contact from "../pages/Contact";

import Login from "../pages/Login";
import Register from "../pages/Register";

import ForgotPassword from "../pages/ForgotPassword";
import VerifyOTP from "../pages/VerifyOTP";
import ResetPassword from "../pages/ResetPassword";
import VerifyRegistrationOTP from "../pages/VerifyRegistrationOTP";

import NotFound from "../pages/NotFound";


// ============================
// USER PAGES
// ============================

import Dashboard from "../pages/Dashboard";
import History from "../pages/History";
import Profile from "../pages/Profile";
import Upload from "../pages/Upload";
import Analytics from "../pages/Analytics";


// ============================
// ADMIN PAGES
// ============================

import AdminLogin from "../pages/AdminLogin";
import AdminDashboard from "../pages/AdminDashboard";
import AdminUsers from "../pages/AdminUsers";
import AdminProfile from "../pages/AdminProfile";


// ============================
// ROUTE PROTECTION
// ============================

import ProtectedRoute from "../components/ProtectedRoute";
import PublicRoute from "../components/PublicRoute";
import AdminProtectedRoute from "../components/AdminProtectedRoute";


// ============================================================
// APP ROUTES
// ============================================================

function AppRoutes() {

  return (

    <BrowserRouter>

      <Routes>

        {/* ================================================== */}
        {/* PUBLIC ROUTES */}
        {/* ================================================== */}

        <Route
          path="/"
          element={<Home />}
        />

        <Route
          path="/about"
          element={<About />}
        />

        <Route
          path="/contact"
          element={<Contact />}
        />


        {/* ================================================== */}
        {/* USER AUTHENTICATION */}
        {/* ================================================== */}

        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />

        <Route
          path="/register"
          element={
            <PublicRoute>
              <Register />
            </PublicRoute>
          }
        />

        <Route
          path="/verify-registration-otp"
          element={<VerifyRegistrationOTP />}
        />

        <Route
          path="/forgot-password"
          element={<ForgotPassword />}
        />

        <Route
          path="/verify-otp"
          element={<VerifyOTP />}
        />

        <Route
          path="/reset-password"
          element={<ResetPassword />}
        />


        {/* ================================================== */}
        {/* USER PROTECTED ROUTES */}
        {/* ================================================== */}

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/upload"
          element={
            <ProtectedRoute>
              <Upload />
            </ProtectedRoute>
          }
        />

        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <History />
            </ProtectedRoute>
          }
        />

        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <Profile />
            </ProtectedRoute>
          }
        />

        <Route
  path="/analytics"
  element={
    <ProtectedRoute>
      <Analytics />
    </ProtectedRoute>
  }
/>


        {/* ================================================== */}
        {/* ADMIN AUTHENTICATION */}
        {/* ================================================== */}

        <Route
          path="/admin/login"
          element={<AdminLogin />}
        />

        <Route
          path="/admin/dashboard"
          element={<AdminDashboard />}
        />

        <Route
          path="/admin/users"
          element={<AdminUsers />}
        />

        <Route
          path="/admin/profile"
          element={<AdminProfile />}
        />

        {/* ================================================== */}
        {/* 404 - NOT FOUND */}
        {/* ================================================== */}

        <Route
          path="*"
          element={<NotFound />}
        />

      </Routes>

    </BrowserRouter>

  );

}


export default AppRoutes;