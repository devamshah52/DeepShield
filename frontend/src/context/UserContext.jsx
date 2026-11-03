import {
  createContext,
  useContext,
  useState,
  useEffect,
} from "react";

import { getDashboard } from "../api/dashboard";


const UserContext = createContext();


export function UserProvider({ children }) {

  const [user, setUser] = useState(null);

  const [loading, setLoading] = useState(true);


  useEffect(() => {

    const loadUser = async () => {

      const token = localStorage.getItem("token");


      // No login token
      if (!token) {

        setUser(null);

        setLoading(false);

        return;

      }


      try {

        const response = await getDashboard();


        // Store the logged-in user's dashboard data
        setUser(response.data);

      } catch (error) {

        console.error(
          "Failed to load user dashboard:",
          error
        );


        // Invalid/expired token
        localStorage.removeItem("token");

        setUser(null);

      } finally {

        setLoading(false);

      }

    };


    loadUser();

  }, []);


  return (

    <UserContext.Provider
      value={{
        user,
        setUser,
        loading,
        isAuthenticated: !!user,
      }}
    >

      {children}

    </UserContext.Provider>

  );

}


export function useUser() {

  return useContext(UserContext);

}