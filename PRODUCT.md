# AROK Construct: product context

## What it is

A record, replay, fork, and diff tool for LLM agent runs, part of the AROK.AI
line (AROK Straylight, AROK Construct). The name comes from Neuromancer's ROM
construct: a recorded mind you can load, question, and run again without
changing the original.

## Who uses it

- **Primary: the engineer who runs the agent.** MSP engineers and IT admins at
  SMBs in regulated-adjacent industries, plus developers building agents for
  them. They open a timeline to answer one question: at which step did this
  run go somewhere it shouldn't have, and what changed going in?
- **Secondary: the reviewer.** A compliance or audit reader who receives the
  timeline file and needs to see, without tooling, what the agent was asked,
  what it answered, and what it did. They may print it.

## Voice

Exacting, quiet, forensic. It states what happened in plain words, shows the
evidence, and never editorializes. A good label reads like a lab notebook
entry, not a dashboard tile.

## Principles

1. **The divergence is the story.** Everything identical recedes; the first
   point of difference and what follows it carry the contrast.
2. **Evidence over summary.** Every claim on screen can be expanded to the
   exact request and response bytes that back it.
3. **Self-contained and silent.** A timeline is an audit artifact: one HTML
   file, no network requests, no fonts or scripts fetched, readable offline in
   ten years, printable.
4. **Plain words for internals.** `cache` is shown as "replayed"; hashes and
   keys appear only as small technical captions.

## Anti-references

- The generic LLM-observability dashboard: dark blue, a grid of metric
  cards, sparklines, hero numbers.
- SaaS-cream with pill badges everywhere.
- Neon-on-black "AI" styling.

## Register

Product. Design serves the task; the brand shows through restraint,
typography, and one deliberate accent.
