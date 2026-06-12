import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Say something and/or attach an image",
    accept_file=True,
    file_type=[".pdf"],):
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    # dublikat
    if prompt.text:
        st.markdown(prompt.text)
    if prompt and prompt["files"]:
        st.chat_message("user").write(prompt["files"][0])
        st.balloons()
    if(prompt.text):      
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        st.chat_message("user").write(prompt.text)
        with st.spinner("Wait for it...", show_time=True):
            response = client.chat.completions.create(
                model=selected_model,
                messages=st.session_state.messages,
                
            )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)