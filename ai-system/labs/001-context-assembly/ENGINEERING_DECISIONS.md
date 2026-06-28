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

Question:

Given retrieved chunks...

How should the system choose context?

Answer:

Policy V1

↓

Take Top-3 by score.