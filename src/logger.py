from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Cấu hình toàn cục (Global Settings)
Settings.llm = Ollama(model="qwen2")
Settings.embed_model = HuggingFaceEmbedding(model_name="keepitreal/vietnamese-sbert")

# 2. Nạp tài liệu và Lập chỉ mục tự động
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents) # Tự động xử lý chunking

# 3. Truy vấn
query_engine = index.as_query_engine()
response = query_engine.query("Điều kiện học bổng là gì?")