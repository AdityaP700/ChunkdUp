import os
import json
import re
from typing import Dict,List,Any
import uuid
from enum import Enum
from datetime import datetime
labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv_path = os.path.join(labs_dir, "data", "conversation.json")
with open(conv_path, "r", encoding="utf-8") as f:
    conversation_lines = json.load(f)

conversation = "\n".join(conversation_lines)
class Decision(Enum):
    STORE = "store"
    UPDATE = "update"
    IGNORE = "ignore"
    MERGE = "merge"
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
                    "importance": 1.0,  # regex = high conf
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                    "source": "conversation",
                    "status": "active"
                }

                if "meta" in rule:
                    for meta_key, group_idx in rule["meta"].items():
                        memory[meta_key] = match.group(group_idx).strip()

                memories.append(memory)

        return memories

class MemoryManager:
    def __init__(self,repository,decision_engine):
        #case of a dependency injection
        #it means the manager relies on the
        # repository to do the heavy lifting of the storage

        self.repository=repository
        self.engine=decision_engine

    def process(self, memory):
        existing = self.repository.find_active_by_key(memory["key"])
        decision = self.engine.decide(existing, memory)
        print(decision)

#the data storage expert
#it manages the CRUD
class MemoryRepository:
    def __init__(self,path):
        self.path=path
    #load the memories
    def load(self):
        #search in the disk ,if it exits or not
        if not os.path.exists(self.path):
            return []
        #if exists ,perform the read operation
        with open(self.path,"r",encoding="utf-8") as f:
            return json.load(f)

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
        if existing_memory is None:
            return Decision.STORE

        if existing_memory.get("key") == new_memory.get("key"):
            if existing_memory.get("value") == new_memory.get("value"):
                return Decision.IGNORE
            else:
                return Decision.UPDATE

        return Decision.STORE

# --- TEST CASES ---
print("\n--- Running DecisionEngine Test Cases ---")
engine = DecisionEngine()

# Case 1
print("Case 1:", engine.decide(None, {"key": "os", "value": "Windows"}).name, "== Expected: STORE")

# Case 2
print("Case 2:", engine.decide({"key": "os", "value": "Windows"}, {"key": "os", "value": "Windows"}).name, "== Expected: IGNORE")

# Case 3
print("Case 3:", engine.decide({"key": "os", "value": "Windows"}, {"key": "os", "value": "Linux"}).name, "== Expected: UPDATE")

# Case 4
print("Case 4:", engine.decide({"key": "os", "value": "Windows"}, {"key": "editor", "value": "VS Code"}).name, "== Expected: STORE")
print("-----------------------------------------\n")

memory_path = os.path.join(labs_dir, "data", "memory.json")
repo = MemoryRepository(memory_path)
manager = MemoryManager(repo)
extractor = MemoryExtractor()

memories = extractor.extract(conversation)

# Process and save each memory
for memory in memories:
    manager.process(memory)

print(f"Successfully processed {len(memories)} memories.")
print("\nAll saved memories in repository:")
print(json.dumps(repo.get_all(), indent=2))