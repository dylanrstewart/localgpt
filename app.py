import streamlit as st
from transformers import pipeline
import torch

# Load your model and tokenizer
@st.cache_resource
def load_model(model_id):
    pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
    return pipe

# Title of the app
st.title("Local GPT")


# Initialize session state to store conversation history and system settings
if "model_id" not in st.session_state:
    st.session_state.model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Input box for the user to provide the model ID
st.subheader("Model Settings")
model_id_input = st.text_input("Enter Model ID", value=st.session_state.model_id, key="default")

# Check if a new model ID is entered and reload the model if needed
if model_id_input != st.session_state.model_id:
    st.session_state.model_id = model_id_input
    pipe = load_model(st.session_state.model_id)
else:
    pipe = load_model(st.session_state.model_id)

if "history" not in st.session_state:
    st.session_state.history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """You are a helpful assistant. Stay on task and do not change the subject of the user's input."""

# Function to interact with Llama model
def generate_response(system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    if st.session_state.history:
        updated_conversation = st.session_state.history + messages
        messages = updated_conversation
    outputs = pipe(messages,
                   max_new_tokens=10_000)
    
    return outputs[0]["generated_text"][-1]["content"]

# Function to log conversation to a text file
def log_conversation():
    conversation = ""
    for message in st.session_state.history:
        if message["role"] == "user":
            conversation += f"You: {message['content']}\n"
        else:
            conversation += f"Assistant: {message['content']}\n"
    return conversation


# Text box for modifying the system prompt
st.subheader("System Settings (Modify Prompt Here)")
system_prompt_input = st.text_area("System Message", value=st.session_state.system_prompt, height=200)
if system_prompt_input != st.session_state.system_prompt:
    st.session_state.system_prompt = system_prompt_input

with st.form("my_form"):
    user_input = st.text_area("User Input", height=400)
    submit_button = st.form_submit_button(label="Send", type="primary")
    if submit_button:
        # Append the user input to history
        st.session_state.history.append({"role": "user", "content": user_input})

        # Generate model response using system prompt
        model_response = generate_response(st.session_state.system_prompt, user_input)

        # Append the model response to history
        st.session_state.history.append({"role": "assistant", "content": model_response})

# Display conversation history
st.write("Conversation History:")
for message in st.session_state.history:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")

# Option to clear the conversation
if st.button("Clear Conversation"):
    st.session_state.history = []


if st.button("Log Conversation"):
    conversation = log_conversation()
    st.download_button(
        label="Download conversation as .txt",
        data=conversation,
        file_name="conversation_log.txt",
        mime="text/plain"
    )
