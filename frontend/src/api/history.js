import API from "./auth";

export const getHistory = () => {

  return API.get(

    "/history/",

    {

      headers: {

        Authorization: `Bearer ${localStorage.getItem("token")}`,

      },

    }

  );

};