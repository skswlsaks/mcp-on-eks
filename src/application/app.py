import streamlit as st
import chat
import utils
import json
# import knowledge_base as kb
import cost_analysis as cost
# import supervisor
# import router
# import swarm
import traceback
import mcp_config

# logging
logger = utils.CreateLogger("MCP")

# title
st.set_page_config(page_title='MCP', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Agent": [
        "Agent를 이용하여 Workflow를 구현합니다."
    ],
    "Agent (Chat)": [
        "Agent를 이용하여 Workflow를 구현합니다. 채팅 히스토리를 이용해 interative한 대화를 즐길 수 있습니다."
    ],
    "Multi-agent Supervisor (Router)": [
        "Multi-agent Supervisor (Router)에 기반한 대화입니다. 여기에서는 Supervisor/Collaborators의 구조를 가지고 있습니다."
    ],
    "LangGraph Supervisor": [
        "LangGraph Supervisor를 이용한 Multi-agent Collaboration입니다. 여기에서는 Supervisor/Collaborators의 구조를 가지고 있습니다."
    ],
    "LangGraph Swarm": [
        "LangGraph Swarm를 이용한 Multi-agent Collaboration입니다. 여기에서는 Agent들 사이에 서로 정보를 교환합니다."
    ],
    "번역하기": [
        "한국어와 영어에 대한 번역을 제공합니다. 한국어로 입력하면 영어로, 영어로 입력하면 한국어로 번역합니다."
    ],
    "문법 검토하기": [
        "영어와 한국어 문법의 문제점을 설명하고, 수정된 결과를 함께 제공합니다."
    ],
    "이미지 분석": [
        "이미지를 업로드하면 이미지의 내용을 요약할 수 있습니다."
    ],
    "비용 분석": [
        "Cloud 사용에 대한 분석을 수행합니다."
    ]
}

def load_image_generator_config():
    config = None
    try:
        with open("image_generator_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"image_generator_config: {config}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
    return config

uploaded_seed_image = None
with st.sidebar:
    st.title("🔮 Menu")

    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다."
        "여기에서는 MCP를 이용해 RAG를 구현하고, Multi agent를 이용해 다양한 기능을 구현할 수 있습니다."
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/mcp)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")

    # radio selection
    mode = "Agent (Chat)"
    st.info(mode)

    # mcp selection
    mcp = ""

    # MCP Config JSON input
    st.subheader("⚙️ MCP Config")

    # Change radio to checkbox
    mcp_options = ["code interpreter", "aws document", "aws cost", "aws cli", "aws cloudwatch", "aws storage", "image generation", "aws diagram", "filesystem", "terminal",  "playwright", "stock data", "stock analysis", "사용자 설정"]
    mcp_selections = {}
    default_selections = ["default", "stock data"]

    with st.expander("MCP 옵션 선택", expanded=True):
        for option in mcp_options:
            default_value = option in default_selections
            mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)

    if not any(mcp_selections.values()):
        mcp_selections["default"] = True

    if mcp_selections["사용자 설정"]:
        mcp_info = st.text_area(
            "MCP 설정을 JSON 형식으로 입력하세요",
            value=mcp,
            height=150
        )
        logger.info(f"mcp_info: {mcp_info}")

        if mcp_info:
            mcp_config.mcp_user_config = json.loads(mcp_info)
            logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")


    mcp = mcp_config.load_selected_config(mcp_selections)
    logger.info(f"mcp: {mcp}")

    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Nova Pro', 'Nova Lite', 'Nova Micro', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=4
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'

    # multi region check box
    select_multiRegion = st.checkbox('Multi Region', value=False)
    multiRegion = 'Enable' if select_multiRegion else 'Disable'

    chat.update(modelName, debugMode, multiRegion, mcp)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # logger.info(f"clear_button: {clear_button}")

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()
    cost.cost_data = {}
    cost.visualizations = {}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []
    uploaded_file = None

    st.session_state.greetings = False
    chat.clear_chat_history()
    st.rerun()


# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")

    with st.chat_message("assistant"):
        sessionState = ""
        response = chat.run_agent(prompt, "Enable", st)
