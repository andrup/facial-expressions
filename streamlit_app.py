import streamlit as st
# import requests
# import base64
from inference_sdk import InferenceHTTPClient

def inference(image):

    bytes_data = image.getvalue()
    # Base64 encode the bytes data
    # bae6s4_encoded_data = base64.b64encode(bytes_data).decode("utf-8")

    # save the image to a file
    with open("image.jpg", "wb") as f:
        f.write(bytes_data)

    API_KEY = st.secrets["API_KEY"]

    try:
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )

        result = CLIENT.infer("image.jpg", model_id="final-emoation/1")
        class_name = result['predictions'][0]["class"]
        confscore = result['predictions'][0]["confidence"]

        result = f"Most likely: {class_name} with {confscore * 100:.1f} % confidence"

    except Exception as e:
        result = f"An error occurred, please contact support."

    return result

def main():
    st.set_page_config(page_title='Women Facial Expression Detection', page_icon=':woman:')
    st.title("Facial Expression Detector")

    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width = 300)
        st.write( inference(uploaded_file) )

if __name__ == "__main__":
    main()
