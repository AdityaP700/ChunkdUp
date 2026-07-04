import os
import json
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

    memories = extractor.extract(conversation)

    # Process and save each memory
    for memory in memories:
        manager.process(memory)

    print(f"Successfully processed {len(memories)} memories.")
    print("\nAll saved memories in repository:")
    print(json.dumps(repo.get_all(), indent=2))