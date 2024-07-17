import streamlit as st
from src.module import load_cache_resources

st.set_page_config("Spam Classifier", page_icon="speech_balloon")
st.title(":speech_balloon: Spam Classifier")
st.caption("The model accuracy is 98.3 %, precision is 100 % and f1 score is 92.2 %")
predictor = load_cache_resources()

message = st.text_input("Enter the message:")
submit = st.button("Submit", type='primary')

if submit:
    with st.spinner("Predicting..."):
        prediction = predictor.predict_class(message)
        st.write(f":blue[Result]: {prediction}")