import os
import streamlit as st
import requests
import json

url = os.environ.get("CHAT_API") if os.environ.get("CHAT_API") else "http://localhost:8000"

title = 'Agentic AI on EKS'
# title
st.set_page_config(page_title=title, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

def get_chat_history():
    user_id = st.session_state.user_id
    response = requests.get(f"{url}/history?user_id={user_id}")
    if response.status_code == 200:
        messages = response.json()["messages"]
        st.session_state.messages = messages
        return messages
    else:
        return []

# Custom CSS for styling tool boxes
st.markdown("""
<style>
    .tool-box {
        background-color: #808080;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .tool-name {
        font-weight: bold;
        color: #4CAF50;
    }
    .tool-input {
        font-family: monospace;
        background-color: #e6e6e6;
        padding: 5px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .tool-result {
        border-top: 1px solid #ccc;
        margin-top: 10px;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🦄 Menu")

    st.markdown(
        "Implementation of an Agentic AI system on Amazon Elastic Kubernetes Service (EKS) utilizing [Strands Agent](https://strandsagents.com/latest/), Agent Tool, and LangGraph frameworks."
    )

    st.subheader("🐱 Chat Configurations")

    st.markdown("🧑‍⚕️ Please input your login name")
    st.text_input("User ID", key="user_id", value="test_user", placeholder="Please input your user ID to load chat history.", )
    st.button("Load Chat History", on_click=get_chat_history)

    # model selection box
    modelName = st.selectbox(
        '🖊️ Please select LLM model',
        ('Nova Pro', 'Nova Lite', 'Nova Micro', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=4
    )

    st.success(f"Connected to {modelName}", icon="💚")

    clear_button = st.button("Reset", key="clear")

st.title("🐳 " + title)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

intro = "Hi! I'm your Stock Analysis Assistant. Tell me what stock information you'd like to analyze and I'll help you."
st.session_state.messages.insert(0, {"role": "assistant", "content": intro})

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []
    uploaded_file = None

    st.session_state.greetings = False
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Please input your message."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        tool_placeholder = st.container()
        full_response = ""
        current_tool = None
        tool_status = {}
        current_expander = None

        response = requests.post(f'{url}/chat', json={"prompt": prompt, "user_id": st.session_state.user_id, "model": modelName}, stream=True)
        st.session_state.tool_progress = {}

        for chunk in response.iter_lines():
            if chunk:
                try:
                    # Try to parse as JSON (for structured tool data)
                    data = json.loads(chunk.decode())

                    if "type" in data:
                        if data["type"] == "message":
                            full_response += data["content"]
                            message_placeholder.markdown(full_response + "▌")

                        elif data["type"] == "tool_start":
                            tool_name = data["name"]
                            tool_status[tool_name] = "running"
                            tool_id = data['tool_id']

                            with tool_placeholder:
                                st.session_state[tool_id] = False
                                current_expander = st.expander(f"🔧 Tool: {tool_name} - {'**Completed** ✅' if st.session_state[tool_id] else '**Running**'}", expanded=True)
                                with current_expander:
                                    st.markdown(f"<div class='running-status'>⏳ Executing tool...</div>", unsafe_allow_html=True)
                                    current_tool = st.empty()

                        elif data["type"] == "tool_input":
                            tool_name = data["name"]
                            tool_input = data["input"]

                            with current_expander:
                                current_tool.markdown(f"""
                                <div class='tool-box'>
                                    <div class='tool-input'>Input parameters: {json.dumps(tool_input, indent=2)}</div>
                                </div>
                                """, unsafe_allow_html=True)

                        elif data["type"] == "tool_result":
                            tool_name = data["name"]
                            tool_result = data["result"]
                            tool_status[tool_name] = "completed"
                            tool_id = data['tool_id']

                            with current_expander:
                                st.session_state[tool_id] = True
                                st.markdown(f"<div class='completed-status'>✅ Tool execution completed</div>", unsafe_allow_html=True)
                                current_tool.markdown(f"""
                                <div class='tool-box'>
                                    <div class='tool-input'>Input parameters: {json.dumps(tool_status.get('input', {}), indent=2)}</div>
                                    <div class='tool-result'>Result: {tool_result}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        # Handle regular text content
                        full_response += data.get("content", "")
                        message_placeholder.markdown(full_response + "▌")

                except json.JSONDecodeError:
                    # Handle plain text chunks
                    text_chunk = chunk.decode()
                    full_response += text_chunk
                    message_placeholder.markdown(full_response + "▌")

        # Final update without cursor
        message_placeholder.markdown(full_response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
