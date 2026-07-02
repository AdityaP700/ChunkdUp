# Lab 002: The Trust Problem (Structured Outputs)

## Core Objective
How can an AI system safely and deterministically consume LLM outputs? In other words, how do we stop treating the LLM like a chatty friend and start treating it like a machine that returns actual, usable data?

## 1. Prompting for Deterministic Structure
We engineered our `PromptBuilder` to explicitly request JSON outputs rather than free-form text. By defining the exact schema we expect, we're basically forcing the LLM into a data-returning straitjacket. 

## 2. Resilient Parsing
Even with strict prompts, LLMs are rebellious and love wrapping their JSON in useless markdown fences. We built an `OutputParser` that acts as the first line of defense:
- It uses regex to ruthlessly strip away markdown formatting.
- If standard parsing fails, it uses aggressive regex (`re.search` with `DOTALL`) to hunt for anything resembling curly braces, salvaging whatever JSON payload it can find from the wreckage.

## 3. Strict Output Validation
Parsing JSON is only half the battle. If the structure is wrong, our app crashes anyway. We built an `OutputValidator` to act as a strict bouncer at the club:
- **Presence Checks:** If you don't have an `answer` or `confidence`, you aren't getting in.
- **Type Safety:** If `citations` isn't a list, get out. 
- **Bounds Checking:** If your `confidence` is somehow `1.5`, you are clearly hallucinating.

## 4. Configurable Trust
We learned a critical production engineering principle: **Never trust the model just because it produced valid JSON.** 
By making validation configurable (`ENABLE_VALIDATION`), we proved that a production system must intentionally reject non-compliant outputs rather than risk poisoning our downstream pipelines with hallucinations. 

Basically, trust no one. Especially the AI.