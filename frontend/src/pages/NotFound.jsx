import { Link } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";
import "../styles/NotFound.css";

function NotFound() {

  return (

    <MainLayout>

      <section className="notfound-page">

        <div className="notfound-card">

          <h1>

            404

          </h1>

          <h2>

            Oops! Page Not Found

          </h2>

          <p>

            The page you're looking for doesn't exist
            or may have been moved.

          </p>

          <Link
            to="/"
            className="home-btn"
          >

            ← Back to Home

          </Link>

        </div>

      </section>

    </MainLayout>

  );

}

export default NotFound;