from datetime import datetime, timedelta

otp_storage = {}


def save_otp(
    email: str,
    otp: str,
    extra_data=None
):

    otp_storage[email] = {

        "otp": otp,

        "expires_at":
            datetime.now() + timedelta(minutes=5),

        "extra_data":
            extra_data

    }


def verify_otp(
    email: str,
    otp: str
):

    if email not in otp_storage:

        return False

    data = otp_storage[email]

    if datetime.now() > data["expires_at"]:

        del otp_storage[email]

        return False

    if data["otp"] != otp:

        return False

    return True


def get_extra_data(
    email: str
):

    if email in otp_storage:

        return otp_storage[email]["extra_data"]

    return None


def remove_otp(
    email: str
):

    if email in otp_storage:

        del otp_storage[email]