import { Link } from "react-router-dom";
import "../styles/Navbar.css";

function Navbar() {
  return (
    <nav className="navbar">

      <div className="logo">
        DeepShield
      </div>

      <ul className="nav-links">
        <li><Link to="/">Home</Link></li>

        <li><Link to="/about">About</Link></li>

        <li><Link to="/contact">Contact</Link></li>
      </ul>

      <div className="auth-buttons">

        <Link to="/login" className="login-btn">
          Login
        </Link>

        <Link to="/register" className="register-btn">
          Register
        </Link>

      </div>

    </nav>
  );
}

export default Navbar;