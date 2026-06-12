import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

folder_name = "zapisane_pliki"

st.sidebar.title("📁 Zarządzanie plikami na serwerze")

if os.path.exists(folder_name):
    pliki = os.listdir(folder_name)
    
    if not pliki:
        st.sidebar.info("Folder jest pusty.")
    else:
        for plik in pliki:
            sciezka_pliku = os.path.join(folder_name, plik)
            
            # Tworzymy ładny wiersz dla każdego pliku
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"📄 {plik}")
            
            # Przycisk do pobrania pliku z serwera na Twój komputer
            with open(sciezka_pliku, "rb") as f:
                col2.download_button(
                    label="Pobierz",
                    data=f,
                    file_name=plik,
                    mime="application/pdf",
                    key=plik # unikalny klucz dla Streamlita
                )
else:
    st.sidebar.info("Brak folderu z zapisanymi plikami.")

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
        # 1. Wyciągamy obiekt pliku (ten, który mi pokazałeś)
        uploaded_file = prompt["files"][0]
        # 2. Tworzymy folder na dysku, jeśli jeszcze nie istnieje
        os.makedirs(folder_name, exist_ok=True)
        # 3. Tworzymy pełną ścieżkę (np. "pobrane_pdf/metaheurystyki lab.pdf")
        file_path = os.path.join("pobrane_pdf", uploaded_file.name)
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