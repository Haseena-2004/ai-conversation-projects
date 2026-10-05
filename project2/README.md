# Project 2: Booking Agent + Eval Harness

**Live demo:** https://superb-llama-82327b.netlify.app |  **PRD:** PRD B in the [PRDs doc](https://claude.ai/code/artifact/78d47c7d-d174-4c3c-8870-934f3894446c)

## Problem
A home-services business uses an AI agent to book appointments. It must handle emergencies, never invent prices or policies, and resist prompt injection, and every prompt change risks breaking something.

## What it does
- A multi-step booking agent (collects name, address, issue, time; confirms a slot).
- Three prompt versions: **v1** baseline, **v2** approved facts + guardrails, **v3** adds read-back confirmation and a human fallback.
- An automated test suite scored by an LLM judge: expected vs actual, pass/fail, **failure category** (hallucination, missed emergency, wrong tone, incomplete info, guardrail breach), latency and estimated cost.

## Results (live demo: 10 scenarios)
| Version | Pass rate | Hallucinations | Avg latency/turn | Est. cost/convo | Failure types |
|---|---|---|---|---|---|
| v1 | 3/10 (30%) | 4/10 | 1.8s | ~$0.0018 | hallucination (4), incomplete info (3) |
| v2 | 10/10 (100%) | 0/10 | 1.4s | ~$0.0021 | none |
| v3 | 10/10 (100%) | 0/10 | 1.4s | ~$0.0025 | none |

## Findings
1. Adding approved business facts and rules (v2) raised pass rate from 30% to 100% and cut hallucinations from 4 to 0. Caveat: v1 had no business facts, so part of the gain is simply giving the agent information.
2. v3 showed no measurable gain over v2 on this suite, at about 19% higher estimated cost. The suite hit a ceiling and cannot separate v2 from v3.
3. Decision: use v2 as the baseline; add harder cases (ambiguous addresses, interruptions, mixed emergencies, longer injection attempts) before deciding whether v3's rules are worth the cost.

## Run the script (20 scenarios)
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python booking_agent_eval.py v1
python booking_agent_eval.py v2
python booking_agent_eval.py v3
```
Each run writes `results_<version>.csv` and prints a summary. Note: the live demo uses 10 of the 20 scenarios, so script results will differ.

## Limitations
The LLM judge can be wrong (spot-check its reasons). Cost is estimated from text length. 10 scenarios is a small suite.

## Next steps
Harder test cases, retrieval for policies, a voice layer (e.g. Vapi or Retell), and adding real failed calls to the suite.
