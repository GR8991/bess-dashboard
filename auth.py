
import streamlit as st


def require_authorized_user():
    """Allow dashboard access only for emails listed in Streamlit Secrets."""

    if "authorized_email" not in st.session_state:
        st.session_state.authorized_email = None

    if st.session_state.authorized_email is None:
        st.title("GR_AIDC profile")
        st.write("Enter your authorized company email to continue.")

        with st.form("email_login"):
            email = st.text_input("Email address")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            email = email.strip().casefold()

            allowed_emails = {
                item.strip().casefold()
                for item in st.secrets["access"]["allowed_emails"]
                if isinstance(item, str) and item.strip()
            }

            if email and email in allowed_emails:
                st.session_state.authorized_email = email
                st.rerun()
            else:
                st.error("Access denied. This email is not authorized.")

        st.stop()

    return st.session_state.authorized_email

def show_logout_button():
    """Display a logout button for the authorized user."""

    with st.sidebar:
        st.divider()
        st.caption(
            f"Signed in as: {st.session_state.get('authorized_email', '')}"
        )

        if st.button("Logout", use_container_width=True):
            st.session_state.authorized_email = None
            st.rerun()

