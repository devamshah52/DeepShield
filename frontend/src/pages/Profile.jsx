import MainLayout from "../layouts/MainLayout";
import { useUser } from "../context/UserContext";

import "../styles/Profile.css";

function Profile() {

  const { user } = useUser();

  const getInitials = (name) => {

    if (!name) return "";

    return name
      .split(" ")
      .map(word => word[0])
      .join("")
      .toUpperCase();

  };

  return (

    <MainLayout>

      <section className="profile-page">

        <div className="profile-card">

          <div className="profile-avatar">

            {getInitials(user?.full_name)}

          </div>

          <h2>

            {user?.full_name}

          </h2>

          <p>

            {user?.email}

          </p>

        </div>


        <div className="profile-info">

          <h3>Account Information</h3>


          <div className="info-row">

            <span>Full Name</span>

            <strong>

              {user?.full_name}

            </strong>

          </div>


          <div className="info-row">

            <span>Email</span>

            <strong>

              {user?.email}

            </strong>

          </div>


          <div className="info-row">

            <span>Total Uploads</span>

            <strong>

              {user?.statistics?.total_uploads ?? 0}

            </strong>

          </div>


          <div className="info-row">

            <span>Genuine Audio</span>

            <strong>

              {user?.statistics?.genuine_audio ?? 0}

            </strong>

          </div>


          <div className="info-row">

            <span>TTS Audio</span>

            <strong>

              {user?.statistics?.tts_audio ?? 0}

            </strong>

          </div>


          <div className="info-row">

            <span>Voice Conversion</span>

            <strong>

              {user?.statistics?.voice_conversion_audio ?? 0}

            </strong>

          </div>

        </div>

      </section>

    </MainLayout>

  );

}

export default Profile;