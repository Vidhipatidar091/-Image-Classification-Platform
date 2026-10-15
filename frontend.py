import streamlit as st
import requests
from PIL import Image

# API configuration
API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Image Classification Platform",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Classification Platform")
st.write("Upload an image to test the classification API.")

# Check API status
if st.button("Check API Status"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            st.success("API is running!")

            if data.get("model_loaded"):
                st.success("Classification model is loaded.")
            else:
                st.warning("API is running, but the model is not loaded yet.")
        else:
            st.error(f"API returned status code {response.status_code}.")

    except requests.RequestException:
        st.error("Could not connect to the API. Check that it is running on port 8001.")

# Image upload
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        try:
            uploaded_file.seek(0)

            response = requests.post(
                f"{API_URL}/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "image/jpeg"
                    )
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction completed!")
                st.write("Predicted class:", result["predicted_class"])
                st.write("Confidence:", result["confidence"])

            elif response.status_code == 503:
                st.warning(
                    "The API is running, but image prediction is unavailable "
                    "because the model has not been loaded."
                )

            else:
                st.error(f"Prediction failed: {response.text}")

        except requests.RequestException as error:
            st.error(f"Could not connect to the API: {error}")