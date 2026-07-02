import os
import json
import re
from typing import Dict,List,Any
import uuid
from datetime import datetime
labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv_path = os.path.join(labs_dir, "data", "conversation.json")
with open(conv_path, "r", encoding="utf-8") as f:
    conversation_lines = json.load(f)

conversation = "\n".join(conversation_lines)
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
    def __init__(self,repository):
        #case of a dependency injection
        #it means the manager relies on the
        # repository to do the heavy lifting of the storage
        self.repository=repository

    def process(self, memory):
        self.repository.add(memory)

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

#it safely returns the current list of saved memories
#allowing the AI to read its own past context
    def get_all(self):
        return self.load()

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