import os
import json
import requests
import streamlit as st

base_url = os.environ.get('BASE_URL', 'http://localhost:8000')
mcp_servers = requests.get(f"{base_url}/get_mcp_servers")
servers = mcp_servers.json()

def change_server_status(server, status):
    servers[server] = not status
    requests.post(f"{base_url}/set_mcp_servers", json=servers)

with st.sidebar:
    st.title('🦙💬 Bedrock Chatbot with MCP ')
    st.write('This chatbot is created using the Amazon Bedrock.')

    container = st.container(border=True)
    container.subheader('MCP Servers')

    for server, status in servers.items():
        on = container.toggle(label=server, value=status, on_change=change_server_status, args=(server, status))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to generate streaming response from Bedrock
def get_bedrock_stream_response(prompt):

    try:
        response = requests.post(f"{base_url}/test_mcpclient", json={"content": prompt})
        yield response.json()['response']

    except Exception as e:
        yield f"Error: {str(e)}"


st.title("MCP Demo UI")


# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Stream the response
        for response_chunk in get_bedrock_stream_response(prompt):
            full_response += response_chunk
            response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

