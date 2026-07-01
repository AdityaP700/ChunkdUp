import os
import json
import re
from typing import Dict,List,Any
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
    "pattern": r"What is ([\w\s]+?)\?",
    "type": "question",
    "key": "topic_asked",
    "value_group": 1
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
                    "type": rule["type"],
                    "key": rule["key"],
                    "value": value,
                    "confidence": 1.0,  # regex = high conf
                    "source": "conversation"
                }

                if "meta" in rule:
                    for meta_key, group_idx in rule["meta"].items():
                        memory[meta_key] = match.group(group_idx).strip()

                memories.append(memory)
                
        return memories

extractor = MemoryExtractor()
memories = extractor.extract(conversation)
print(memories)