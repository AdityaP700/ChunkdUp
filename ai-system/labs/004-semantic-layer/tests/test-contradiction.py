import os
import sys
import json
from datetime import datetime, timezone
import uuid

# Setup paths
tests_dir = os.path.dirname(os.path.abspath(__file__))
semantic_layer_dir = os.path.dirname(tests_dir)
labs_dir = os.path.dirname(semantic_layer_dir)

# Add semantic layer and experiments to path
if semantic_layer_dir not in sys.path:
    sys.path.append(semantic_layer_dir)
experiments_dir = os.path.join(semantic_layer_dir, "experiments")
if experiments_dir not in sys.path:
    sys.path.append(experiments_dir)

from retrieval_class import MemoryRetriever
from prompt_builder import PromptBuilder
from experiment_001 import MemoryRepository, MemoryManager, DecisionEngine, MemoryScorer

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

def create_memory(key, value, type):
    return {
        "id": str(uuid.uuid4()),
        "key": key,
        "value": value,
        "type": type,
        "frequency": 1,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "conversation"
    }

def test_contradiction():
    print("--- Test 3: Contradiction ---")
    
    # 1. Setup Repo and Lifecycle Manager
    temp_repo_path = os.path.join(tests_dir, "temp_repo.json")
    if os.path.exists(temp_repo_path):
        os.remove(temp_repo_path)
        
    repo = MemoryRepository(temp_repo_path)
    engine = DecisionEngine()
    scorer = MemoryScorer()
    manager = MemoryManager(repo, engine, scorer)
    
    # 2. Simulate User Lifecycle
    print("\nSimulating Lifecycle...")
    print("User: 'I use Neovim'")
    mem1 = create_memory("editor", "Neovim", "tool")
    manager.process(mem1)
    
    print("User: 'Actually, I prefer VS Code'")
    mem2 = create_memory("editor", "VS Code", "tool")
    manager.process(mem2)
    
    # 3. Setup Pipeline
    retriever = MemoryRetriever(repo)
    builder = PromptBuilder()
    llm = LLMCaller(provider="gemini")
    parser = OutputParser()
    validator = OutputValidator(raise_on_fail=False)
    
    # 4. Query
    query = "Which editor do I use?"
    print(f"\nQuery: {query}")
    
    # 5. Execute Pipeline
    results = retriever.retrieve(query, k=1)
    
    # Quick sanity check on retrieval
    retrieved_value = results[0]["value"] if results else "None"
    print(f"Retriever selected: {retrieved_value}")
    
    prompt = builder.build(query=query, contexts={"memories": results, "documents": []}, variant="expert")
    
    print("\nCalling LLM...")
    raw_response = llm.generate(prompt)
    parsed_dict = parser.parse(raw_response)
    validated_dict = validator.validate(parsed_dict)
    
    # 6. Output
    print("\n=== ACTUAL VALIDATED JSON ===")
    print(json.dumps(validated_dict, indent=2))
    
    # Assertions
    answer = validated_dict.get("answer", "").lower()
    
    if "vs code" in answer or "visual studio code" in answer:
        print("\n[PASSED]: LLM correctly identified VS Code as the current editor.")
    else:
        print("\n[FAILED]: LLM hallucinated or returned the old editor (Neovim).")
        
    if "neovim" not in answer:
        print("[PASSED]: The outdated memory (Neovim) was successfully excluded.")
    else:
        print("[FAILED]: The outdated memory was included in the answer.")
        
    # Store the results in the data dir
    results_dir = os.path.join(labs_dir, "data", "tests-lab4")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "contradiction_memory.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(validated_dict, f, indent=2)
    print(f"\nResult saved to: {result_path}")
    
    # Cleanup
    if os.path.exists(temp_repo_path):
        os.remove(temp_repo_path)

if __name__ == "__main__":
    test_contradiction()
