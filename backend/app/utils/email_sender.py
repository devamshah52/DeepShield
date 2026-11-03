import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import EMAIL_ADDRESS, EMAIL_PASSWORD


def send_otp_email(receiver_email: str, otp: str):

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Email configuration is missing.")
        return False

    subject = "DeepShield Password Reset OTP"

    body = f"""
Hello,

Your DeepShield password reset OTP is:

{otp}

This OTP is valid for 5 minutes.

If you did not request a password reset, please ignore this email.

Regards,
DeepShield Team
"""

    message = MIMEMultipart()

    message["From"] = EMAIL_ADDRESS
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(
        MIMEText(body, "plain")
    )

    try:

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                EMAIL_ADDRESS,
                EMAIL_PASSWORD
            )

            server.sendmail(
                EMAIL_ADDRESS,
                receiver_email,
                message.as_string()
            )

        return True

    except Exception as e:

        print("Email Error:", e)

        return False