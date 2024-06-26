from openai import OpenAI
import streamlit as st
import tiktoken
import json
import datetime
import uuid

st.title("CareerBuddy")

# Define the maximum number of tokens allowed
MAX_TOKENS = 10

enc = tiktoken.encoding_for_model("gpt-4o")

if "b_openai_api_key" not in st.session_state:
    st.session_state["b_openai_api_key"] = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "token_limit_exceeded" not in st.session_state:
    st.session_state["token_limit_exceeded"] = False

# Define the function to count tokens
def count_tokens(messages, model):
    num_tokens = num_tokens_from_messages(messages, model)
    return num_tokens

def reset_app():
    st.session_state["openai_api_key"] = ""
    st.session_state.messages = []

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" or "gpt-4o" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

with st.sidebar:
    openai_api_key = st.text_input('OpenAI API Key', type='password', key='openai_api_key')

    if st.button('Enter'):
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter a valid OpenAI API key!', icon='⚠')
        else:
            if "client" not in st.session_state:
                st.session_state["client"] = OpenAI(api_key=openai_api_key)

    if st.button('Trial'):
        if "client" not in st.session_state:
            st.session_state["client"] = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        st.session_state["b_openai_api_key"] = True
        # add a token counter later to limit the number of tokens used per session

    st.button('Reset', on_click=reset_app)

if st.session_state["b_openai_api_key"] == True and st.session_state["token_limit_exceeded"] == False:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = st.session_state["client"].chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

if 'messages' in st.session_state:
    messages = st.session_state.messages
    num_tokens = count_tokens(messages, st.session_state["openai_model"])
    
    # Check if the token count has reached the maximum allowed
    if num_tokens >= MAX_TOKENS:
        st.warning("Token limit exceeded! ({} tokens)".format(num_tokens))
        st.session_state["token_limit_exceeded"] = True
    else:
        st.session_state["token_limit_exceeded"] = False