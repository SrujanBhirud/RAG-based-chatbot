import streamlit as st
import requests

# Set up the page
st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("LLM Chatbot for Books")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat function
def clear_chat():
    st.session_state.chat_history = []

# Create a sidebar with a clear chat button
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        clear_chat()
        st.rerun()  # Refresh the app to reflect changes
        
st.subheader("Chat History")
chat_container = st.container()

# User input field
query = st.text_input("Ask a question about books:")

# Function to query backend
def ask_chatbot(user_query):
    try:
        with st.spinner("Thinking..."):
            response = requests.post("http://localhost:8000/ask", json={"query": user_query}, timeout=12)
            
            # Ensure we got a valid response
            if response.status_code == 200:
                response_data = response.json()
                
                if isinstance(response_data, dict) and "response" in response_data:
                    return response_data["response"]
                else:
                    return "⚠️ Unexpected response format from chatbot."
            else:
                return f"⚠️ Error: Received status code {response.status_code}"

    except requests.exceptions.RequestException as e:
        return f"⚠️ Error: Unable to connect to the chatbot. ({e})"

# Handle user input and update chat history
if query:
    response = ask_chatbot(query)
    
    # Store query-response in session state
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Chatbot", response))

# Display chat history

with chat_container:
    for role, text in st.session_state.chat_history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.markdown(f"**{role}:** {text}")
