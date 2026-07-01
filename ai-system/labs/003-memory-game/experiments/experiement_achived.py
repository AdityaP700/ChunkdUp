conversation = """
User: I'm building ChunkdUp in Python.
User: I prefer concise explanations.
User: I use Windows.
User: My favorite editor is VS Code.
User: What is cosine similarity?
"""
class MemoryExtractor:

    def extract(self, conversation: str):
        memories=[]

        if "I'm building" in conversation:
            project= conversation.split("I'm building ")[1].split("in")[0].strip()
            memories.append({
                "key":"project",
                "value":project
            })
        if "I use" in conversation:
            os = conversation.split("I use")[1].split(".")[0].strip()
            memories.append({
                "key": "os",
                "value": os
            })

        # Rule 3
        if "I prefer" in conversation:
            preference = conversation.split("I prefer")[1].split(".")[0].strip()
            memories.append({
                "key": "preference",
                "value": preference
            })

        if "My favorite" in conversation:
            editor = conversation.split("My favorite")[1].split(".")[0].strip()
            memories.append({
                "key": "editor",
                "value": editor
            })
        return memories

extractor=MemoryExtractor()
memories=extractor.extract(conversation)
print(memories)