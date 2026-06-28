# Lab 001 - Context Assembly

## Engineering Question

Given retrieved chunks, how should an AI system decide what reaches the LLM?

## Today's Goal

Understand the structure of retrieved chunks before implementing any context assembly logic.

## Hypothesis

A context assembler should operate only on retrieved chunks.
It should not care how they were retrieved.

## Status

✅ Retrieved chunks successfully loaded and inspected.

## Next Step

Implement the simplest possible ContextAssembler that selects the Top-3 chunks.