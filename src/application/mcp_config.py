import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

mcp_user_config = {}

def load_selected_config(mcp_selections: dict[str, bool]):
    #logger.info(f"mcp_selections: {mcp_selections}")
    loaded_config = {}

    # True 값만 가진 키들을 리스트로 변환
    selected_servers = [server for server, is_selected in mcp_selections.items() if is_selected]
    logger.info(f"selected_servers: {selected_servers}")

    with open(f"mcp_config.json", "r") as f:
        config = json.loads(f.read())
        for server in selected_servers:
            logger.info(f"server: {server}")
            if config[server]:
                loaded_config.update(config[server]["mcpServers"])

    logger.info(f"loaded_config: {loaded_config}")

    return {
        "mcpServers": loaded_config
    }
