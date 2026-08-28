import streamlit as st
from src.firestore_db import upsert_user
import requests
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw

def make_circular_profile_image(image_url: str) -> str:
    """
    Download a profile image, crop it to a square,
    make it circular, and return it as a base64
    PNG data URI suitable for an HTML <img src="">.
    """

    response = requests.get(
        image_url,
        timeout=10,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    # Open downloaded image
    image = Image.open(
        BytesIO(response.content)
    ).convert("RGBA")

    # Crop to a centered square
    image = ImageOps.fit(
        image,
        (96, 96),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

    # Create circular transparency mask
    mask = Image.new(
        "L",
        (96, 96),
        0
    )

    draw = ImageDraw.Draw(mask)

    draw.ellipse(
        (0, 0, 95, 95),
        fill=255
    )

    # Apply circular mask
    circular_image = Image.new(
        "RGBA",
        (96, 96),
        (255, 255, 255, 0)
    )

    circular_image.paste(
        image,
        (0, 0),
        mask
    )

    # Convert to PNG bytes
    output = BytesIO()

    circular_image.save(
        output,
        format="PNG"
    )

    # Convert to base64
    encoded_image = base64.b64encode(
        output.getvalue()
    ).decode("utf-8")

    # Return an HTML-safe data URI
    return f"data:image/png;base64,{encoded_image}"

def require_login():
    """Require Google authentication before showing the app."""

    # User is not logged in
    if not st.user.is_logged_in:
        st.title("🔐 Login Required")
        st.write("Please sign in with Google to continue.")

        if st.button("🔑 Login with Google"):
            st.login()

        st.stop()


    # Safely display available user information
    user = st.user
    user_id = getattr(user, "sub", None)

    if not user_id:
        st.error(
            "Unable to determine your authenticated Google identity."
        )
        st.stop()

    user_email = getattr(user, "email", "")
    user_name = getattr(user, "name", "")
    user_picture = getattr(user, "picture", "")

    upsert_user(
        user_id=user_id,
        email=user_email,
        display_name=user_name,
        profile_picture=user_picture,
    )

    st.session_state.collab_user_id = user_id

    # 1. Grab Google's profile picture and convert it
    #    into a circular base64 PNG for app.py.
    profile_pic_url = getattr(user, "picture", "")

    if profile_pic_url:
        try:
            st.session_state.user_picture = (
                make_circular_profile_image(
                    profile_pic_url
                )
            )
        except Exception:
            st.session_state.user_picture = ""
    else:
        st.session_state.user_picture = ""    
