import streamlit as st
import requests
import time

FLASK_URL = "http://localhost:5001"

st.title("Question Generator Chatbot")

uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

questions_placeholder = st.empty()
success_message_placeholder = st.empty()

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)

    if st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            files = {'file': uploaded_file}
            response = requests.post(f"{FLASK_URL}/upload", files=files)

            if response.status_code == 200:
                st.session_state['questions'] = response.json().get('questions', [])
                success_message_placeholder.success("Questions generated successfully!")
                time.sleep(2)  # Wait for 1 second
                success_message_placeholder.empty()  
            else:
                st.error("Error generating questions!")

if st.session_state['questions']:
    questions_placeholder.header("Generated Questions")
    formatted_questions = "\n\n".join(st.session_state['questions'])  # Two newlines for extra spacing
    st.text_area("Generated Questions", value=formatted_questions, height=300, disabled=True)

if st.session_state['questions']:
    question_numbers = [f"Question {i + 1}" for i in range(len(st.session_state['questions']))]
    
    selected_question_index = st.selectbox("Select a question to answer", question_numbers)
    selected_index = int(selected_question_index.split(' ')[-1]) - 1  # Extract the number and adjust for 0-based index
    selected_question = st.session_state['questions'][selected_index]

    user_answer = st.text_area("Your Answer", height=150) 

    if st.button("Verify Answer"):
        with st.spinner("Verifying answer..."):
            response = requests.post(
                f"{FLASK_URL}/verify",
                json={"question": selected_question, "answer": user_answer}
            )
            
            if response.status_code == 200:
                verification_result = response.json().get('verification', '')
                st.success(verification_result)
            else:
                st.error("Error verifying the answer!")

if st.button("Reset"):
    st.session_state.clear()
    st.success("Session reset. Please upload a new document.")
