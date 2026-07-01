import os
import json
import sys
import re
from typing import Dict,Any,List
from sentence_transformers import SentenceTransformer, util

# Load .env file manually from the parent directory of experiments
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip()

def load_chunks():
    #first locate the labs dir
    #second the chunk inside the chunks.json
    #open up in the way it will encode in utf 8
    labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chunks_path = os.path.join(labs_dir, "data", "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks()

class SemanticRetriever:
    def __init__(self, chunks):
        # store the chunks
        self.chunks = chunks
        # load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Build a list containing the "text" from every chunk.
        chunk_texts=[chunk["text"] for chunk in chunks]
        self.chunk_embeddings = self.model.encode(chunk_texts)
        # print(self.chunk_embeddings.shape)


    def retrieve(self, query: str = "", k: int = 10):
        if not self.chunks:
            return []
        # encode query
        query_embedding = self.model.encode(query)
        # compare with stored embeddings
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]

        # Zip scores with chunks
        scored_chunks = []
        for idx, score in enumerate(cos_scores):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            scored_chunks.append(chunk)

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:k]

class ContextAssembler:
    def __init__(self):
        self.context_budget=800

#it doesnt takes any data ,it just takes the chunks
#input when assemble() is called
    def assemble(self,chunks:list[dict])->list[dict]:

        if not chunks:
            print("No chunks provided to assembler.")
            return []

        sorted_chunks=sorted(
            chunks,
            #konsa cheez ko sort out karna hai
            #kis tarike se karna hai
            #aur kya reverse hoga??nhi
            key=lambda x : x.get("score",0.0),
            reverse=True
        )

        selected = []
        current_length = 0
        budget = self.context_budget

        print(f"\n=== Context Budget: {budget} characters ===")

        for i, chunk in enumerate(sorted_chunks, 1):
            chunk_text = chunk.get("text", "")
            added_length = len(chunk_text)          # characters
            word_count = len(chunk_text.split())    # for readability

            if current_length + added_length > budget:
                print(f"[Skip] Skipped Chunk {i} ({word_count} words, {added_length} chars) "
                      f"-> Budget exceeded")
                continue  # V3: skip and try next

            # It fits -> add it
            selected.append(chunk)
            current_length += added_length
            remaining = budget - current_length

            print(f"[Add] Added Chunk {i} ({word_count} words, {added_length} chars) "
                  f"-> Remaining Budget: {remaining} chars")

        print(f"\nFinal Selection: {len(selected)} chunks "
              f"(used {current_length} / {budget} chars)\n")

        return selected

class PromptBuilder:

    def build(self, query: str, selected_chunks: list[dict], variant: str = "basic") -> str:
        if not selected_chunks:
            context_str = "No relevant context found."
        else:
            lines = [f"{i}. {c.get('text', '').strip()}" for i, c in enumerate(selected_chunks, 1)]
            context_str = "\n".join(lines)

        if variant == "basic":
            return f"""You are a helpful AI assistant.

Answer the question using ONLY the provided context.
Return JSON without confidence:
{{
  "answer": "your answer here",
  "citations": <list of integer chunk numbers (e.g. [1, 6]) used to answer the question>
}}

Context:
----------------
{context_str}
----------------

Question:
{query}

Answer:"""

        elif variant == "expert":
            return f"""You are an expert AI assistant.

Use ONLY the provided context.
If the answer is not present, the "answer" field should be "I don't know."
Always be concise and accurate.
Return only the answer in a valid JSON object:
{{
  "answer": "your answer here"
}}

Context:
----------------
{context_str}
----------------

Question:
{query}

Answer:"""

        else:
            return self.build(query, selected_chunks, "basic")

class LLMCaller:
    """Simple wrapper for Gemini or Claude."""

    def __init__(self, provider: str = "gemini"):  # "gemini" or "claude"
        self.provider = provider.lower()

        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEYS"))
            self.model = genai.GenerativeModel("gemini-2.5-flash")  # or gemini-1.5-pro
        elif self.provider == "claude":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEYS"))
        else:
            raise ValueError("Provider must be 'gemini' or 'claude'")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text.strip()
        else:  # claude
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # or latest
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

class OutputParser:
    """it has one responsibility"""
    """Raw string->python dictionary"""
    def parse(self,response : str):
        """receives a string that might be wrapped in json"""
        #strips markdown fences ,loads json ,returns dict
        if not response:
            return{}

        cleaned = re.sub(r'^```json\s*|\s*```$', '',response.strip(),flags=re.MULTILINE)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
                #this extract the largest block of text enclosed in curly braces
                #even if that text spans across multiple lines
                #re.DOTALL changes this behavior. It forces the dot (.)
                # to match absolutely anything, including newlines.
            match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"could not parse json from response:{e}")
class OutputValidator:
    """takes a python dict->validates schemas and types -> """
    """validated dict or raises"""
    def __init__(self,raise_on_fail:bool=True):
        self.raise_on_fail=raise_on_fail

    def validate(self,data:Dict[str,Any])->Dict[str,Any]:
        checks=[
            self._check_answer_present,
            self._check_confidence_present,
            self._check_confidence_is_float,
            self._check_confidence_in_range,
            self._check_citations_is_list
        ]
        for check in checks:
            passed,msg=check(data)
            if not passed :
                if self.raise_on_fail:
                    raise ValueError(f"Validation failed:{msg}")
                return {}
        return data

    def _check_answer_present(self,data:Dict)->tuple[bool,str]:
        if "answer" not in data:
            return False, "missing required key 'answer'"
        return True, ""

    def _check_confidence_present(self, data: Dict) -> tuple[bool, str]:
        if "confidence" not in data:
            return False, "missing required key 'confidence'"
        return True, ""

    def _check_confidence_is_float(self, data: Dict) -> tuple[bool, str]:
        if not isinstance(data.get("confidence"), (float, int)):
            return False, f"'confidence' must be float or int, got {type(data.get('confidence')).__name__}"
        return True, ""

    def _check_confidence_in_range(self, data: Dict) -> tuple[bool, str]:
        conf = data.get("confidence")
        if not (0 <= conf <= 1):
            return False, f"'confidence' must be between 0 and 1, got {conf}"
        return True, ""

    def _check_citations_is_list(self, data: Dict) -> tuple[bool, str]:
        if "citations" in data and not isinstance(data.get("citations"), list):
            return False, f"'citations' must be a list, got {type(data.get('citations')).__name__}"
        return True, ""

question ="Should we always trust highest scoring chunks??"

retriever = SemanticRetriever(chunks)
assembler = ContextAssembler()
builder=PromptBuilder()

# Configuration Flag (Option 1)
ENABLE_VALIDATION = True

retrieved_chunks = retriever.retrieve(query=question,k=10)
final_context = assembler.assemble(retrieved_chunks)

# print(f"Received {len(retrieved_chunks)} chunks from the retrieval")
# print(f"Selected {len(final_context)} chunks for the model\n")

# for i, chunk in enumerate(final_context,1):
#     print(f"{i}.[scores={chunk['score']}] {chunk['text']}")


prompt_a = builder.build(question, final_context, variant="basic")
prompt_b = builder.build(question, final_context, variant="expert")

# print("\n=== PROMPT A (Basic) ===")
# print(prompt_a)
# print("\n=== PROMPT B (Expert) ===")
# print(prompt_b)

# Run with LLM
llm = LLMCaller(provider="gemini")   # Change to "claude" anytime
# print("\n=== LLM RESPONSE (Prompt B) ===")
answer = llm.generate(prompt_b)
# print(answer)

parser = OutputParser()
parsed_dict = parser.parse(answer)

print("\n=== PARSED DICTIONARY ===")
print(parsed_dict)

if ENABLE_VALIDATION:
    validator = OutputValidator(raise_on_fail=True)
    try:
        validated_dict = validator.validate(parsed_dict)
        print("\n=== VALIDATION SUCCESS ===")
        print("clean data:", validated_dict)
    except ValueError as e:
        print("\n=== VALIDATION FAILED ===")
        print("bad output from llm:", e)