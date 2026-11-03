import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

/* ===========================
   LOGIN
=========================== */

export const loginUser = (userData) => {

  const formData = new URLSearchParams();

  formData.append(
    "username",
    userData.email
  );

  formData.append(
    "password",
    userData.password
  );

  return API.post(
    "/login",
    formData,
    {
      headers: {
        "Content-Type":
          "application/x-www-form-urlencoded",
      },
    }
  );

};


/* ===========================
   REGISTER (SEND OTP)
=========================== */

export const sendRegistrationOTP = (userData) => {

  return API.post(
    "/register/send-otp",
    {

      full_name:
        userData.full_name,

      email:
        userData.email,

      password:
        userData.password,

    }
  );

};


/* ===========================
   VERIFY REGISTRATION OTP
=========================== */

export const verifyRegistrationOTP = (
  email,
  otp
) => {

  return API.post(
    "/register/verify-otp",
    {

      email,

      otp,

    }
  );

};


/* ===========================
   FORGOT PASSWORD
=========================== */

export const forgotPassword = (email) => {

  return API.post(
    "/forgot-password",
    {
      email,
    }
  );

};


/* ===========================
   VERIFY OTP (Forgot Password)
=========================== */

export const verifyOTP = (
  email,
  otp
) => {

  return API.post(
    "/verify-otp",
    {

      email,

      otp,

    }
  );

};


/* ===========================
   RESET PASSWORD
=========================== */

export const resetPassword = (
  email,
  otp,
  newPassword
) => {

  return API.post(
    "/reset-password",
    {

      email,

      otp,

      new_password:
        newPassword,

    }
  );

};

/* ===========================
   AUDIO UPLOAD
=========================== */

export const uploadAudio = async (file) => {

  const formData = new FormData();

  formData.append(
    "audio",
    file
  );

  return API.post(

    "/upload",

    formData,

    {

      headers: {

        Authorization: `Bearer ${localStorage.getItem("token")}`,

        "Content-Type": "multipart/form-data",

      },

    }

  );

};

export default API;