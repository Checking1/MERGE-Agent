import os

# DeepSeek API config (environment variables override defaults)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-a2c1e72f1264417bbda9af56a3e1ccb5")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.8"))
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "500"))

# Neo4j connection params (aligned with src/util/graph_db_util.py)
NEO4J_PARAMS = {
    "host": "localhost",
    "port": 7687,
    "user": "neo4j",
    "password": "12345678",
    "name": "neo4j",
}
