# Project 1: Call Intelligence + Next Best Action

**Live demo:** https://cosmic-dodol-00ea00.netlify.app/ |  **PRD:** PRD A in the [PRDs doc](https://claude.ai/code/artifact/78d47c7d-d174-4c3c-8870-934f3894446c)

## Problem
Teams running thousands of sales, renewal and collections calls cannot review them all. Angry or legal-threat calls get missed, and AI-drafted follow-up messages can promise things that were never discussed.

## What it does
Call transcript (English or Hinglish) in, structured JSON out:
- intent, customer stage, sentiment, objection
- **escalation flag** with a reason
- **confidence score** and a **needs-human-review** flag
- **next best action**: channel, timing and a draft message in the customer's language
- a **rule-based check** that flags numbers or offer words in the message that are not in the call

## Results (live demo: 3 calls, labelled by me before running)
The 3 calls: a price objection (English), an EMI promise (Hinglish), and an angry customer threatening to report.

| Metric | Result |
|---|---|
| Escalation agreement | 3/3 (100%) |
| Intent agreement | 3/3 (100%) |
| Objection agreement | 2/3 (67%) |
| Channel agreement | 1/3 (33%) |
| Avg confidence when intent right / wrong | 0.82 / n/a (no wrong intents, so calibration is untested) |
| Risky messages flagged | 0 |
| Calls routed to human review | 1 of 3 |

## Findings
1. Channel choice is the weakest part (1 of 3 agreed with my labels). Next: write explicit channel rules and test on more calls.
2. Escalation worked on the hostile call (regulator threat), with a clear reason, and the model recommended no follow-up.
3. The model drafted a Hinglish message for an English-speaking customer, and the message check is keyword-based, so it would miss subtler unsupported claims.

## Run the script (larger test)
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python call_intelligence_nba.py generate 10     # N synthetic calls (default 10)
python call_intelligence_nba.py extract         # writes extractions.csv and a blank labels.csv
# Read transcripts.json and fill labels.csv by hand BEFORE looking at extractions.csv
python call_intelligence_nba.py score           # agreement, confidence check, insights
```

## Limitations
3 calls is a demo, not proof of accuracy. Transcripts are synthetic. The model and I are both Claude-based, so agreement may look better than against an independent labeller.

## Next steps
Test on 30+ calls, add explicit channel rules, replace the keyword check with a claim-level check, use real conversion outcomes to rank actions.
