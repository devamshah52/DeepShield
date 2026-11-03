import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import "./styles/Global.css";

import App from "./App.jsx";

import { UserProvider } from "./context/UserContext";

createRoot(document.getElementById("root")).render(

  <StrictMode>

    <UserProvider>

      <App />

    </UserProvider>

    <ToastContainer
      position="top-right"
      autoClose={3000}
      hideProgressBar={false}
      newestOnTop={true}
      closeOnClick
      pauseOnHover
      draggable
      theme="colored"
    />

  </StrictMode>

);