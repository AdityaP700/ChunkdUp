import os
import sys
import json

# Setup paths
tests_dir = os.path.dirname(os.path.abspath(__file__))
semantic_layer_dir = os.path.dirname(tests_dir)
labs_dir = os.path.dirname(semantic_layer_dir)

# Add semantic layer to path so we can import its modules
if semantic_layer_dir not in sys.path:
    sys.path.append(semantic_layer_dir)

from retrieval_class import MemoryRetriever
from prompt_builder import PromptBuilder

# Load .env file
env_path = os.path.join(labs_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip()

# Import Lab 2 classes dynamically
import importlib.util
lab2_file = os.path.join(labs_dir, "002-structured-outputs", "experiments", "experiment_001.py")
spec = importlib.util.spec_from_file_location("lab2_experiment", lab2_file)
lab2_module = importlib.util.module_from_spec(spec)
sys.modules["lab2_experiment"] = lab2_module
spec.loader.exec_module(lab2_module)

LLMCaller = lab2_module.LLMCaller
OutputParser = lab2_module.OutputParser
OutputValidator = lab2_module.OutputValidator

class MockRepo:
    def __init__(self, memories):
        self.memories = memories
    def get_all(self):
        return self.memories

def test_multiple_memories():
    print("--- Test 2: Multiple Memories ---")
    
    # 1. Setup Repo with mock data (Linux, Neovim, ChunkdUp)
    mock_memory = [
        {"key": "editor", "value": "Neovim", "type": "tool", "status": "active"},
        {"key": "os", "value": "Linux", "type": "environment", "status": "active"},
        {"key": "project_name", "value": "ChunkdUp", "type": "project", "status": "active"}
    ]
    repo = MockRepo(mock_memory)
    
    # 2. Setup Pipeline
    retriever = MemoryRetriever(repo)
    builder = PromptBuilder()
    llm = LLMCaller(provider="gemini")
    parser = OutputParser()
    validator = OutputValidator(raise_on_fail=False)
    
    # 3. Query
    query = "What development setup do I use and what project am I building?"
    print(f"Query: {query}")
    
    # 4. Execute Pipeline
    # Retrieve top 3 to capture all relevant mock data
    results = retriever.retrieve(query, k=3)
    prompt = builder.build(query=query, contexts={"memories": results, "documents": []}, variant="expert")
    
    print("\nCalling LLM...")
    raw_response = llm.generate(prompt)
    parsed_dict = parser.parse(raw_response)
    validated_dict = validator.validate(parsed_dict)
    
    # 5. Output
    print("\n=== ACTUAL VALIDATED JSON ===")
    print(json.dumps(validated_dict, indent=2))
    
    # Assertions
    answer = validated_dict.get("answer", "").lower()
    citations = validated_dict.get("citations", [])
    
    if "neovim" in answer and "linux" in answer and "chunkdup" in answer:
        print("\n[PASSED]: LLM successfully synthesized multiple memories.")
    else:
        print("\n[FAILED]: LLM failed to synthesize all memories.")
        
    if len(citations) > 0:
        print("[PASSED]: Citations were provided as expected.")
    else:
        print("[FAILED]: Citations were empty.")
        
    # Store the results in the data dir
    results_dir = os.path.join(labs_dir, "data", "tests-lab4")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "multiple_memories.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(validated_dict, f, indent=2)
    print(f"\nResult saved to: {result_path}")

if __name__ == "__main__":
    test_multiple_memories()
