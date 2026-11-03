import API from "./auth";


export const getAnalytics = () => {

  return API.get(

    "/analytics/",

    {

      headers: {

        Authorization: `Bearer ${localStorage.getItem("token")}`,

      },

    }

  );

};