
import streamlit as st


def require_authorized_user():
    """Require a verified identity and explicit email authorization."""

    if not st.user.is_logged_in:
        st.title("Wind-BESS Intelligence")
        st.write("Sign in using an approved account.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Sign in with Google", use_container_width=True):
                st.login("google")

        with col2:
            if st.button("Sign in with Microsoft", use_container_width=True):
                st.login("microsoft")

        st.stop()

    email = st.user.get("email", "")
    verified = st.user.get("email_verified", None)

    if not isinstance(email, str) or not email.strip():
        st.error("No valid email identity was provided.")
        st.stop()

    if verified is not True:
        st.error("A verified email identity is required.")
        st.stop()

    email = email.strip().casefold()

    allowed = {
        item.strip().casefold()
        for item in st.secrets["access"]["allowed_emails"]
        if isinstance(item, str) and item.strip()
    }

    if email not in allowed:
        st.error("Access denied. Contact the application administrator.")
        st.stop()

    return email
