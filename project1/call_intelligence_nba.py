"""
Project 1: Call Intelligence + Next Best Action (India EdTech / EMI collections)
Steps:
  python call_intelligence_nba.py generate [N] -> transcripts.json (N synthetic calls, default 10, English + Hinglish)
  python call_intelligence_nba.py extract    -> extractions.csv + labels.csv (blank, fill it yourself)
  (open labels.csv, read each transcript, fill the human_* columns for every row, save)
  python call_intelligence_nba.py score      -> agreement % + product insights
Needs: pip install anthropic ; export ANTHROPIC_API_KEY=...
"""
import anthropic, csv, json, sys
from collections import Counter

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

OUTCOMES = ["interested and ready to pay", "price objection, says fees are too high", "asks to call back later",
            "wrong number", "angry and hostile", "says already paid", "needs parent's approval first",
            "comparing with a competitor", "trust issue, doubts course quality", "EMI due, promises to pay Friday"]
CONTEXTS = ["EdTech course renewal (JEE/NEET prep)", "Bank EMI payment reminder"]
LANGS = ["English", "Hinglish (Hindi-English mix, Roman script)"]

INTENTS = ["interested", "not_interested", "callback", "wrong_number", "already_paid", "will_pay_later"]
OBJECTIONS = ["price", "timing", "trust", "competitor", "parent_approval", "none"]
CHANNELS = ["call", "whatsapp", "sms", "none"]

STAGES = ["new_lead", "considering", "negotiating", "committed", "at_risk", "lost"]
EXTRACT_SYSTEM = f"""You analyse a customer call transcript for a B2C company. Return JSON only:
{{"intent": one of {INTENTS}, "customer_stage": one of {STAGES}, "sentiment": "positive|neutral|negative",
"objection": one of {OBJECTIONS}, "escalate": true|false, "escalation_reason": "short or empty",
"confidence": number 0-1, "needs_human_review": true|false,
"next_action": {{"channel": one of {CHANNELS}, "timing": "e.g. within 1 hour / tomorrow 6pm / none",
"message": "short draft message in the customer's language, or empty"}}}}
Rules: escalate=true for complaint/legal threats, abuse or repeated unresolved complaints. needs_human_review=true if confidence < 0.7 or the call is ambiguous. Never offer discounts or terms not mentioned in the call. For wrong_number or hostile customers use channel "none"."""


def ask(system, user, max_tokens=700):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": user}])
    return r.content[0].text


def parse_json(text):
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def generate(n=None):
    n = n or (int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    data = []
    for i in range(n):
        outcome, ctx, lang = OUTCOMES[i % len(OUTCOMES)], CONTEXTS[i % 2], LANGS[(i // 2) % 2]
        t = ask("You write realistic, slightly messy call-centre transcripts. Output only the dialogue as 'Agent:' / 'Customer:' lines, 8-12 lines.",
                f"Context: {ctx}. Language: {lang}. Customer outcome: {outcome}. Do not state the outcome label explicitly.")
        data.append({"id": i + 1, "transcript": t})
        print("generated", i + 1)
    json.dump(data, open("transcripts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


KEYS = ["id", "intent", "customer_stage", "sentiment", "objection", "escalate", "escalation_reason", "confidence",
        "needs_human_review", "channel", "timing", "message", "message_risk"]


def message_risk(msg, transcript):
    import re
    t, m, flags = transcript.lower(), (msg or "").lower(), []
    for n in re.findall(r"\d[\d,.%]*", m):
        if n.rstrip(".,") not in t:
            flags.append(f"number {n} not in call")
    for w in ["discount", "offer", "waive", "free", "cashback"]:
        if w in m and w not in t:
            flags.append(f"'{w}' not in call")
    return "; ".join(flags)


def extract():
    data = json.load(open("transcripts.json", encoding="utf-8"))
    rows = []
    for d in data:
        row = {k: "" for k in KEYS}
        row["id"] = d["id"]
        try:
            j = parse_json(ask(EXTRACT_SYSTEM, d["transcript"]))
            na = j["next_action"]
            row.update(intent=j["intent"], customer_stage=j["customer_stage"], sentiment=j["sentiment"],
                       objection=j["objection"], escalate=j["escalate"], escalation_reason=j["escalation_reason"],
                       confidence=j["confidence"], needs_human_review=bool(j["needs_human_review"]) or float(j["confidence"]) < 0.7,
                       channel=na["channel"], timing=na["timing"], message=na["message"],
                       message_risk=message_risk(na["message"], d["transcript"]))
        except Exception as e:  # keep failures: they are part of your quality analysis
            row.update(intent="PARSE_ERROR", message=str(e), needs_human_review=True)
        rows.append(row)
        print("extracted", d["id"])
    with open("extractions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEYS); w.writeheader(); w.writerows(rows)
    with open("labels.csv", "w", newline="", encoding="utf-8") as f:  # read transcripts.json, fill every row by hand
        w = csv.writer(f)
        w.writerow(["id", "human_intent", "human_objection", "human_channel", "human_escalate"])
        for d in data:
            w.writerow([d["id"], "", "", "", ""])
    print(f"\nAllowed values -> intent: {INTENTS}\nobjection: {OBJECTIONS}\nchannel: {CHANNELS}\nescalate: yes/no")


def score():
    ex = {int(r["id"]): r for r in csv.DictReader(open("extractions.csv", encoding="utf-8"))}
    labels = [r for r in csv.DictReader(open("labels.csv", encoding="utf-8")) if r["human_intent"].strip()]
    esc = lambda r: "yes" if str(r["escalate"]).lower() == "true" else "no"
    checks = [("intent", lambda r: r["intent"], "human_intent"), ("objection", lambda r: r["objection"], "human_objection"),
              ("channel", lambda r: r["channel"], "human_channel"), ("escalation", esc, "human_escalate")]
    for name, get, col in checks:
        hits = sum(get(ex[int(l["id"])]) == l[col].strip() for l in labels)
        print(f"{name:11s} agreement: {hits}/{len(labels)} = {hits/len(labels):.0%}")
    conf = lambda l: float(ex[int(l["id"])]["confidence"] or 0)
    right = [conf(l) for l in labels if ex[int(l["id"])]["intent"] == l["human_intent"].strip()]
    wrong = [conf(l) for l in labels if ex[int(l["id"])]["intent"] != l["human_intent"].strip()]
    avg = lambda a: f"{sum(a)/len(a):.2f}" if a else "n/a"
    print(f"Avg confidence when intent right: {avg(right)} | when wrong: {avg(wrong)}")
    rows = list(ex.values())
    print(f"JSON parse failures: {sum(r['intent'] == 'PARSE_ERROR' for r in rows)}/{len(rows)}")
    print(f"Escalations flagged: {sum(esc(r) == 'yes' for r in rows)} | Needs human review: {sum(str(r['needs_human_review']) == 'True' for r in rows)} | Risky NBA messages: {sum(bool(r['message_risk']) for r in rows)}")
    print("\n== Insights (all calls) ==")
    print("Intent mix:   ", dict(Counter(r["intent"] for r in rows)))
    print("Stage mix:    ", dict(Counter(r["customer_stage"] for r in rows)))
    print("Objections:   ", dict(Counter(r["objection"] for r in rows if r["objection"] != "none")))
    print("Next channel: ", dict(Counter(r["channel"] for r in rows)))
    print("\nNext: read rows where the model disagreed with you, write 3 product insights + 1 PRD.")


if __name__ == "__main__":
    {"generate": generate, "extract": extract, "score": score}[sys.argv[1]]()
