# https://blog.futuresmart.ai/building-a-conversational-voice-chatbot-integrating-openais-speech-to-text-text-to-speech
# Reference: See above

import streamlit as st
# import argparse

import os
from helpers import text_to_speech, autoplay_audio, speech_to_text, speech_to_text_speechrecognition
from generate_answer import base_model_chatbot, with_pdf_chatbot
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

def main(answer_mode: str):
    # Float feature initialization
    float_init()

    def initialize_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm RESI! How can I help you?"}
            ]


    initialize_session_state()

    st.image("./img/logo.png", width=300)


    footer_container = st.container()
    with footer_container:
        col1, col2, col3, a,b,c,d, col5, col6= st.columns(9)  # Adjust the ratio as needed

        with col1:
            st.text("1.")
        with col2:
            st.image("./img/tap.png", width=45)
        with col3:
            audio_bytes = audio_recorder(text="")

        with col5:
            st.text("2.")
        with col6:
            st.image("./img/talk.png", width=45)
            
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if audio_bytes:
        # Write the audio bytes to a file
        #with st.spinner("Transcribing..."):
        with st.spinner(""):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.write(transcript)
            os.remove(webm_file_path)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            #with st.spinner("Thinking🤔..."):
            with st.spinner("🤔"):
                if answer_mode == 'base_model':
                 final_response = base_model_chatbot(st.session_state.messages)
                elif answer_mode == 'pdf_chat':
                    print('--------->', st.session_state.messages)
                    final_response = with_pdf_chatbot(st.session_state.messages)
            #with st.spinner("Generating audio response..."):
            with st.spinner("🤔"):
                audio_file = text_to_speech(final_response)
                autoplay_audio(audio_file)
            st.write(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            os.remove(audio_file)

            if final_response == '🔓':
                    st.dialog("🔓✅")

    footer_container.float("bottom: 3rem;")
 
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run OpenAI Conversational Chatbot")
    # parser.add_argument('--answer_mode', type=str, default='base_model',
    #                     choices=['base_model', 'pdf_chat'],
    #                     help="Specify the answer mode ('base_model' or 'pdf_chat')")

    # args = parser.parse_args()

    st.set_page_config(page_title='Resi.ai', page_icon = "./img/penguin.png", initial_sidebar_state = 'auto')

    main(answer_mode='base_model') # Or: answer_mode='base_model'