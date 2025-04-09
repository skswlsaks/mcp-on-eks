import json
import requests
import streamlit as st

with st.sidebar:
    st.title('🦙💬 Bedrock Chatbot with MCP ')
    st.write('This chatbot is created using the Amazon Bedrock.')

    container = st.container(border=True)
    container.subheader('Model Context Protocol')
    on = container.toggle("Activate feature1")
    on = container.toggle("Activate feature2")
    on = container.toggle("Activate feature3")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to generate streaming response from Bedrock
def get_bedrock_stream_response(prompt):

    try:
        # response = requests.post("http://localhost:8000/test_mcpclient", json={"content": prompt}, stream=True)

        # # Iterate through the streaming response
        # for line in response.iter_lines(decode_unicode=True):
        #     if line:
        #         line_response = json.loads(line)
        #         if line_response['content']:
        #             yield line_response['content']
        response = requests.post("http://localhost:8000/test_mcpclient", json={"content": prompt})
        yield response.json()['response']

    except Exception as e:
        yield f"Error: {str(e)}"


st.title("MCP Demo")
res = requests.get("http://localhost:8000")

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

# run with
# streamlit run main.py