import os
import json
import math
import re
# from typing import Dict,List,Any
import uuid
from enum import Enum
from datetime import datetime, timezone
import sys
labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv_path = os.path.join(labs_dir, "data", "conversation.json")
with open(conv_path, "r", encoding="utf-8") as f:
    conversation_lines = json.load(f)

conversation = "\n".join(conversation_lines)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policies import Decision, PolicyFactory
# Load .env file manually from the parent directory of experiments
labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(labs_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip()
class MemoryExtractor:
    def __init__(self):
        #rules : patterns ,types ,key,value_extractor
        self.rules=[
            {
                "pattern": r"I'm building ([\w\d\-]+) in ([\w\d\+#]+)",
                "type": "project",
                "key": "project_name",
                "value_group": 1,
                "meta": {"language": 2}

            },
            {
                "pattern": r"I prefer ([\w\s]+?)\.",
                "type": "preference",
                "key": "response_style",
                "value_group": 1
            },
            {
                "pattern": r"I use (Windows|Mac|Linux)",
                "type": "environment",
                "key": "os",
                "value_group": 1
            },
            {
                "pattern": r"I use (Python|Java|C\+\+|Rust|Go)",
                "type": "environment",
                "key": "programming_language",
                "value_group": 1
            },
            {
                "pattern": r"My favorite editor is ([\w\s]+?)\.",
                "type": "tool",
                "key": "editor",
                "value_group": 1
            }
        ]
    def extract(self, conversation: str):
        memories = []
        for rule in self.rules:
            matches = re.finditer(rule["pattern"], conversation, re.IGNORECASE)

            for match in matches:
                value = match.group(rule["value_group"]).strip()
                memory = {
                    "id": str(uuid.uuid4()),
                    "type": rule["type"],
                    "key": rule["key"],
                    "value": value,
                    "frequency": 1,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source": "conversation",
                    "status": "active"
                }

                if "meta" in rule:
                    for meta_key, group_idx in rule["meta"].items():
                        memory[meta_key] = match.group(group_idx).strip()

                memories.append(memory)

        return memories

class MemoryScorer:
    def score(self, memory):
        scores = {
            "project": 1.0,
            "preference": 0.9,
            "environment": 0.8,
            "tool": 0.7,
            "question": 0.3
        }
        return scores.get(memory.get("type"), 0.5)

class MemoryRanker:
    def __init__(self, scorer):
        self.scorer = scorer

    def rank(self, memories):
        now = datetime.now(timezone.utc)
        ranked = []
        for mem in memories:
            # 1. Semantic Importance
            semantic = self.scorer.score(mem)

            # 2. Frequency (Logarithmic cap, arbitrary simple math for the test)
            freq = mem.get("frequency", 1)
            freq_score = math.log10(freq + 9) - 1  # 1 freq = 0 bonus. 100 freq = 1.03 bonus.

            # 3. Recency (Inverse decay based on days)
            updated_str = mem.get("updated_at")
            if updated_str:
                # Handle ISO format with Z
                updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                days_old = max(0, (now - updated_at).total_seconds() / 86400)
            else:
                days_old = 30 # fallback

            recency_score = 1.0 / (days_old + 1.0)

            # 4. Composite Score
            composite = semantic + freq_score + recency_score

            # Injecting for debugging in the test output
            mem["_ranking_debug"] = {
                "semantic": round(semantic, 3),
                "frequency_bonus": round(freq_score, 3),
                "recency_bonus": round(recency_score, 3),
                "composite_score": round(composite, 3),
                "days_old": round(days_old, 1)
            }
            ranked.append((composite, mem))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in ranked]

class MemoryManager:
    def __init__(self,repository,decision_engine,scorer):
        #case of a dependency injection
        #it means the manager relies on the
        # repository to do the heavy lifting of the storage

        self.repository=repository
        self.engine=decision_engine
        self.scorer=scorer
        self.threshold=0.6

    def process(self, memory):
        importance = self.scorer.score(memory)
        if importance < self.threshold:
            print(f"Discarded memory: {memory.get('value')} (type: {memory.get('type')}, score: {importance})")
            return
        memory["importance"] = importance
        existing = self.repository.find_active_by_key(memory["key"])
        decision = self.engine.decide(existing, memory)

        #now since we have the existing and the prev memory
        #our job is to decide what we need to do right now
        #either we can store ,ignore or update
        if decision==Decision.STORE:
            self.repository.add(memory)

        elif decision == Decision.IGNORE:
             print(f"Ignored duplicate memory: {memory['key']}")

    #added a new decision ,the decision merge to increase the freq
        elif decision == Decision.MERGE:
             self.repository.merge(existing, memory)
             print(f"Merged duplicate memory (increased frequency): {memory['key']}")

        elif decision == Decision.UPDATE:
             self.repository.update(memory)

#the data storage expert
#it manages the CRUD
class MemoryRepository:
    def __init__(self,path):
        self.path=path
    #load the memories
    def load(self):
        #search in the disk ,if it exits or not
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return []
        #if exists ,perform the read operation
        with open(self.path,"r",encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def save(self, memories):
        with open(self.path, "w", encoding="utf-8") as f:
            #we are saving the data in a targeted file
            #where there is an indentation of 4
            #where after a certain space the line breaks

             json.dump(memories, f, indent=4)
    #core transactional pipeline
    #managing CRUD
    def add(self,memory):
        #either it could read
        memories = self.load()
        #or it could modify
        memories.append(memory)
        #it could write
        self.save(memories)

    #we are taking the new memory since we are updatig
    #so the logic is
    #for the same key ,why we cant simply update the value right??\
    #for when its updated_at ?? we can simply tune it with the datetime()
    #then call the method to save the memories
    def update(self, new_memory):
        memories = self.load()
        for mem in memories:
            ## this solves the real problem
            ## the key value pair issue
            #if for example the key and the status is active
            #just go for updating the value
            if mem.get("key") == new_memory.get("key") and mem.get("status") == "active":
                for k, v in new_memory.items():
                    if k not in ["id", "created_at", "frequency"]:
                        mem[k] = v
                mem["frequency"] = 1
                mem["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                break
        self.save(memories)
  # we are updating the frequency based upon same key and the status
    def merge(self, existing_memory, new_memory):
        memories = self.load()
        for mem in memories:
            if mem.get("key") == existing_memory.get("key") and mem.get("status") == "active":
                mem["frequency"] = mem.get("frequency", 1) + 1
                mem["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                break
        self.save(memories)

#we need this
#the reason is simple,its that lets suppose we get diff values for same keys
#then how we will know if there are any existing memory or not
#the work of the manager is to manage
#not to search
#hence we need this implementation
    def find_active_by_key(self, key):
        memories = self.load()
        for memory in memories:
            if memory.get("key") == key and memory.get("status") == "active":
                return memory
        return None

#it safely returns the current list of saved memories
#allowing the AI to read its own past context
    def get_all(self):
        return self.load()

class DecisionEngine:
    def decide(self, existing_memory, new_memory):
        policy = PolicyFactory.get(new_memory["type"])
        if not policy:
            return Decision.STORE
        return policy.decide(existing_memory, new_memory)

if __name__ == "__main__":
    memory_path = os.path.join(labs_dir, "data", "memory.json")
    repo = MemoryRepository(memory_path)
    engine = DecisionEngine()
    scorer = MemoryScorer()
    manager = MemoryManager(repo, engine, scorer)
    extractor = MemoryExtractor()

    # --- Wire up retrieval testing ---
    # First, let's load retrieval class and prompt builder
    from retrieval_class import MemoryRetriever
    from prompt_builder import PromptBuilder
    import importlib.util

    # Let's dynamically import Lab 2 classes because of the filename collision (both are experiment_001.py)
    lab2_file = os.path.join(labs_dir, "002-structured-outputs", "experiments", "experiment_001.py")
    spec = importlib.util.spec_from_file_location("lab2_experiment", lab2_file)
    lab2_module = importlib.util.module_from_spec(spec)
    sys.modules["lab2_experiment"] = lab2_module
    spec.loader.exec_module(lab2_module)

    LLMCaller = lab2_module.LLMCaller
    OutputParser = lab2_module.OutputParser
    OutputValidator = lab2_module.OutputValidator

    retriever = MemoryRetriever(repo)
    builder = PromptBuilder()
    llm = LLMCaller(provider="gemini")
    parser = OutputParser()
    validator = OutputValidator(raise_on_fail=False) # don't crash the script on fail

    # Inject mock memory into repository directly for the test case
    mock_memory = [
        {"key": "editor", "value": "Neovim", "type": "tool", "status": "active"},
        {"key": "os", "value": "Linux", "type": "environment", "status": "active"},
        {"key": "project_name", "value": "ChunkdUp", "type": "project", "status": "active"}
    ]

    # Let's save the mock memory so retriever can get it via repo.get_all()
    repo.save(mock_memory)

    print("\n--- Test 1 ---")
    query1 = "Which editor do I use?"
    print(f"Query: {query1}\n")
    results1 = retriever.retrieve(query1, k=1)

    # Build prompt
    prompt1 = builder.build(
        query=query1,
        contexts={
            "memories": results1,
            "documents": []
        },
        variant="expert"
    )

    print("Calling LLM...")
    raw_response1 = llm.generate(prompt1)
    parsed_dict1 = parser.parse(raw_response1)
    validated_dict1 = validator.validate(parsed_dict1)
    print("=== VALIDATED JSON (Test 1) ===")
    print(json.dumps(validated_dict1, indent=2))
    print("===============================\n")

    print("\n--- Test 2 ---")
    query2 = "What project am I building?"
    print(f"Query: {query2}\n")
    results2 = retriever.retrieve(query2, k=1)

    # Build prompt
    prompt2 = builder.build(
        query=query2,
        contexts={
            "memories": results2,
            "documents": []
        },
        variant="expert"
    )

    print("Calling LLM...")
    raw_response2 = llm.generate(prompt2)
    parsed_dict2 = parser.parse(raw_response2)
    validated_dict2 = validator.validate(parsed_dict2)
    print("=== VALIDATED JSON (Test 2) ===")
    print(json.dumps(validated_dict2, indent=2))
    print("===============================\n")