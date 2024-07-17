import streamlit as st
from src.module import load_cache_resources

st.set_page_config("Spam Classifier")
st.title("Spam Classifier")
predictor = load_cache_resources()

message = st.text_input("Enter the message:")
submit = st.button("Submit", type='primary')

if submit:
    with st.spinner("Predicting..."):
        prediction = predictor.predict_class(message)
        st.write(f":blue[Result]: {prediction}")