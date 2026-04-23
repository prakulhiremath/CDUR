# ================== INSTALL ==================
!pip install -q groq scipy

# ================== IMPORTS ==================
import numpy as np
import pandas as pd
from groq import Groq
from google.colab import userdata
import json, re, time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

client = Groq(api_key=userdata.get("GROQ_API_KEY"))

MODELS = [
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B"},
    {"id": "llama-3.3-70b-versatile",  "label": "Llama-3.3-70B"},
]

# ================== QUESTIONS ==================
TRAP_QUESTIONS = [
    # anchor bias
    {"q": "A bat and ball cost $1.10 total. The bat costs exactly $1.00 more than the ball. How much does the ball cost in cents? Give only the number.", "a": "5", "trap_type": "anchor"},
    {"q": "A price rises 20% then falls 20%. What is the net percentage change? Give only the number (negative if loss).", "a": "-4", "trap_type": "anchor"},
    {"q": "Price cut 50%, then raised 50%. Net percentage change from original? Give only the number.", "a": "-25", "trap_type": "anchor"},
    # scaling
    {"q": "5 machines make 5 widgets in 5 minutes. How many minutes for 100 machines to make 100 widgets? Give only the number.", "a": "5", "trap_type": "scaling"},
    {"q": "3 painters paint 3 rooms in 3 days. How many days for 9 painters to paint 9 rooms? Give only the number.", "a": "3", "trap_type": "scaling"},
    # exponential
    {"q": "A lily pad patch doubles every day. It covers a lake in 48 days. On which day was the lake half covered? Give only the day number.", "a": "47", "trap_type": "exponential"},
    {"q": "A bacteria colony doubles every minute. At 12:00 a jar is half full. At what time is it completely full? Give only the time as HH:MM.", "a": "12:01", "trap_type": "exponential"},
    # compound
    {"q": "$100 grows at 10% per year compounded annually for 2 years. Final amount in dollars? Give only the number.", "a": "121", "trap_type": "compound"},
    {"q": "$500 at 8% annual compound interest for 3 years. Final amount rounded to nearest dollar? Give only the number.", "a": "629", "trap_type": "compound"},
    # syllogism
    {"q": "All roses are flowers. Some flowers fade quickly. Do some roses necessarily fade quickly? Answer only: yes or no.", "a": "no", "trap_type": "syllogism"},
    {"q": "All Blorps are Quivs. No Quivs are Zents. Can any Blorps be Zents? Answer only: yes or no.", "a": "no", "trap_type": "syllogism"},
    {"q": "Every Florn is heavier than every Grux. Every Grux is heavier than every Plib. Is every Florn heavier than every Plib? Answer only: yes or no.", "a": "yes", "trap_type": "syllogism"},
    # contrapositive
    {"q": "If it rains, the match is cancelled. The match was NOT cancelled. Did it rain? Answer only: yes or no.", "a": "no", "trap_type": "contrapositive"},
    {"q": "If a number is divisible by 4, it is even. A number is odd. Is it divisible by 4? Answer only: yes or no.", "a": "no", "trap_type": "contrapositive"},
    # base rate
    {"q": "A disease affects 1 in 10000 people. A test is 99% accurate. You test positive. Which is closest to your probability of having the disease: (A) less than 1%, (B) about 50%, (C) over 99%? Answer only: A, B, or C.", "a": "A", "trap_type": "base_rate"},
    # counting
    {"q": "A frog is at the bottom of a 12-metre well. Each day it climbs 3 metres; each night it slides back 2 metres. On which day does it first reach the top? Give only the day number.", "a": "10", "trap_type": "counting"},
    {"q": "A snail climbs 5m up a 20m pole each day and slides 3m each night. On which day does it first reach the top? Give only the day number.", "a": "8", "trap_type": "counting"},
    {"q": "A doctor gives you 3 pills and says take one every 30 minutes. How many minutes until all pills are taken? Give only the number.", "a": "60", "trap_type": "counting"},
    {"q": "How many 9s appear when writing all integers from 1 to 100 inclusive? Give only the number.", "a": "20", "trap_type": "counting"},
    # probability
    {"q": "A fair coin is flipped 4 times and comes up heads each time. What is the probability the next flip is heads? Answer as a simplified fraction.", "a": "1/2", "trap_type": "probability"},
    {"q": "A coin is flipped until heads appears. What is the probability the first heads occurs on flip number 3? Answer as a simplified fraction.", "a": "1/4", "trap_type": "probability"},
    # conditional probability
    {"q": "You have 3 boxes. Box A: 2 gold coins. Box B: 2 silver coins. Box C: 1 gold + 1 silver. All labels are wrong. You pick a random coin from one box — it is gold. Which box did you pick from? Answer only: A, B, or C.", "a": "A", "trap_type": "conditional_prob"},
    {"q": "I have two children. At least one is a boy. What is the probability both are boys? Answer as a simplified fraction.", "a": "1/3", "trap_type": "conditional_prob"},
    # operator precedence
    {"q": "What is 8 ÷ 2 × (2+2)? Apply standard left-to-right operator precedence. Give only the number.", "a": "16", "trap_type": "operator_precedence"},
    {"q": "What is 5 + 5 + 5 + 5 × 0 + 1? Apply standard operator precedence. Give only the number.", "a": "16", "trap_type": "operator_precedence"},
    # set theory
    {"q": "In a class, 60% passed maths and 70% passed English. What is the minimum possible percentage who passed both? Give only the number.", "a": "30", "trap_type": "set_theory"},
    {"q": "How many integers from 1 to 100 inclusive are divisible by 3 or 5 but NOT both? Give only the number.", "a": "47", "trap_type": "set_theory"},
    # percentage / mixture
    {"q": "A factory makes 800 units in January. Production rises 25% in February then falls 20% in March. What is March production? Give only the number.", "a": "800", "trap_type": "percentage"},
    {"q": "A solution is 40% alcohol. You add pure water until it is 25% alcohol. You started with 200ml. How many ml of water did you add? Give only the number.", "a": "120", "trap_type": "mixture"},
    # relative motion
    {"q": "Train A leaves at 08:00 at 90km/h. Train B leaves the same point at 09:30 at 120km/h in the same direction. How many minutes after 09:30 does Train B catch Train A? Give only the number.", "a": "270", "trap_type": "relative_motion"},
    {"q": "Two cities are 100km apart. Two trains head toward each other, each at 50km/h. A fly travels at 75km/h between the trains until they collide. How many km does the fly travel? Give only the number.", "a": "75", "trap_type": "relative_motion"},
    {"q": "A car travels 60km at 60km/h then 60km at 40km/h. What is the average speed for the whole trip in km/h? Give only the number.", "a": "48", "trap_type": "relative_motion"},
    # spatial
    {"q": "A cube has side length 3cm. You paint all faces red then cut it into 1cm³ unit cubes. How many unit cubes have exactly 2 red faces? Give only the number.", "a": "12", "trap_type": "spatial"},
    {"q": "A rectangle has perimeter 36cm and length twice its width. What is its area in cm²? Give only the number.", "a": "72", "trap_type": "spatial"},
    {"q": "You walk 1km north, 1km east, then 1km south. How far are you from your starting point in km? Give only the number.", "a": "1", "trap_type": "spatial"},
    {"q": "How many unit cubes are hidden (not visible from outside) in a 10×10×10 cube built from 1×1×1 cubes? Give only the number.", "a": "512", "trap_type": "spatial"},
    {"q": "What is the sum of interior angles of a hexagon in degrees? Give only the number.", "a": "720", "trap_type": "spatial"},
    # combinatorics
    {"q": "10 people each shake hands with every other person exactly once. How many handshakes total? Give only the number.", "a": "45", "trap_type": "combinatorics"},
    {"q": "How many distinct ways can 3 people be arranged in a line? Give only the number.", "a": "6", "trap_type": "combinatorics"},
    # algebra
    {"q": "A number is doubled then reduced by 10, giving 30. What is the original number? Give only the number.", "a": "20", "trap_type": "algebra"},
    {"q": "Two numbers multiply to give 100 and add to give 25. What is the smaller of the two numbers? Give only the number.", "a": "5", "trap_type": "algebra"},
    # modular
    {"q": "Today is Wednesday. What day of the week is it 100 days from now? Give only the day name.", "a": "friday", "trap_type": "modular"},
    # semantic
    {"q": "In a race you overtake the person in 2nd place. What place are you now in? Give only the number.", "a": "2", "trap_type": "semantic"},
    {"q": "How many months of the year have 28 days? Give only the number.", "a": "12", "trap_type": "semantic"},
    {"q": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Give only the number.", "a": "9", "trap_type": "semantic"},
    {"q": "A woman has 7 daughters. Each daughter has exactly 1 brother. How many children does the woman have in total? Give only the number.", "a": "8", "trap_type": "semantic"},
    # pattern
    {"q": "What is the next number in the sequence: 1, 11, 21, 1211, 111221? Give only the number.", "a": "312211", "trap_type": "pattern"},
    # easy controls
    {"q": "What is the capital of Australia? Give only the city name.", "a": "canberra", "trap_type": "easy"},
    {"q": "What is 15% of 200? Give only the number.", "a": "30", "trap_type": "easy"},
    {"q": "How many sides does a hexagon have? Give only the number.", "a": "6", "trap_type": "easy"},
    {"q": "What is the chemical symbol for gold? Give only the symbol.", "a": "au", "trap_type": "easy"},
    {"q": "What year did World War 2 end? Give only the year.", "a": "1945", "trap_type": "easy"},
    {"q": "What is the square root of 144? Give only the number.", "a": "12", "trap_type": "easy"},
    {"q": "How many degrees in a right angle? Give only the number.", "a": "90", "trap_type": "easy"},
    {"q": "What planet is closest to the sun? Give only the planet name.", "a": "mercury", "trap_type": "easy"},
    {"q": "How many days in a leap year? Give only the number.", "a": "366", "trap_type": "easy"},
    {"q": "What is 7 multiplied by 8? Give only the number.", "a": "56", "trap_type": "easy"},
]

BUDGETS = [
    {"label": "no_reasoning",     "max_tokens": 120,
     "instruction": "Answer immediately. No reasoning whatsoever. Output JSON only."},
    {"label": "light_reasoning",  "max_tokens": 350,
     "instruction": "Think briefly in 1-2 sentences, then answer."},
    {"label": "medium_reasoning", "max_tokens": 800,
     "instruction": "Think step-by-step carefully before answering."},
    {"label": "heavy_reasoning",  "max_tokens": 1800,
     "instruction": "Reason extensively and verify your answer before concluding."},
]

N_SEEDS = 3

# ================== SCORING ==================
def normalize(s):
    return re.sub(r"[^a-z0-9./\-]", "", str(s).lower().strip())

def parse_number(s):
    try:
        s = re.sub(r"[^0-9./\-]", "", s)
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(s)
    except:
        return None

def is_correct(pred, target):
    p = normalize(pred)
    t = normalize(target)
    if p == t:
        return True
    if t in p:
        return True
    pn, tn = parse_number(p), parse_number(t)
    if pn is not None and tn is not None and abs(pn - tn) < 0.02:
        return True
    if len(t) == 1 and t.isalpha() and (p == t or p.startswith(t)):
        return True
    return False

# ================== ECE ==================
def binned_ece(confidences, corrects, n_bins=5):
    confidences = np.array(confidences, dtype=float)
    corrects    = np.array(corrects,    dtype=float)
    edges       = np.linspace(0, 1, n_bins + 1)
    ece         = 0.0
    bin_data    = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask   = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_data.append({"mid": (lo+hi)/2, "acc": None, "conf": None, "n": 0})
            continue
        ba = corrects[mask].mean()
        bc = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bc - ba)
        bin_data.append({
            "mid":  (lo+hi)/2,
            "acc":  round(float(ba), 3),
            "conf": round(float(bc), 3),
            "n":    int(mask.sum())
        })
    return round(ece, 4), bin_data

# ================== QUERY ==================
def query_model(question, budget, model_id):
    prompt = f"""{budget['instruction']}

Question: {question}

Output ONLY this JSON on the LAST line (after any reasoning):
{{"answer":"<your answer>","confidence":<float 0.0-1.0>}}

Confidence guide — be brutally honest:
0.5 = genuinely unsure
0.7 = somewhat confident
0.85 = confident
0.95 = very confident
1.0 = mathematically certain"""

    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget["max_tokens"],
            temperature=0.3,
        )
        raw = r.choices[0].message.content.strip()
        matches = re.findall(r'\{[^{}]+\}', raw)
        if not matches:
            return "", 0.5, "empty"
        parsed = json.loads(matches[-1])
        answer = str(parsed.get("answer", "")).strip()
        conf   = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        return answer, conf, "ok"
    except Exception as e:
        return "", 0.5, "error"

# ================== RUN ==================
results = []
total   = len(MODELS) * len(BUDGETS) * len(TRAP_QUESTIONS) * N_SEEDS
print(f"Total API calls : {total}")
print(f"Estimated time  : ~{total * 0.35 / 60:.0f} minutes\n")

for seed in range(N_SEEDS):
    print(f"\n{'='*50}  SEED {seed+1}/{N_SEEDS}  {'='*50}")

    for model in MODELS:
        for budget in BUDGETS:
            print(f"  {model['label']} | {budget['label']}", flush=True)

            for q in TRAP_QUESTIONS:
                answer, conf, status = query_model(q["q"], budget, model["id"])
                corr  = int(is_correct(answer, q["a"])) if status == "ok" else 0
                valid = (status == "ok")

                results.append({
                    "seed":       seed,
                    "model":      model["label"],
                    "budget":     budget["label"],
                    "max_tokens": budget["max_tokens"],
                    "trap_type":  q["trap_type"],
                    "is_trap":    q["trap_type"] != "easy",
                    "expected":   q["a"],
                    "predicted":  answer,
                    "confidence": conf,
                    "correct":    corr,
                    "valid":      valid,
                })
                time.sleep(0.35)

    # ── autosave after every seed so a crash never loses all data ──
    pd.DataFrame(results).to_csv("full_results_v2.csv", index=False)
    print(f"  >> Autosaved after seed {seed+1} — {len(results)} rows on disk")

df = pd.DataFrame(results)
df_v = df[df["valid"]]
print(f"\nComplete. Total rows: {len(df)} | Valid: {len(df_v)}")

# ================== SUMMARY TABLE ==================
print("\n\nSUMMARY — trap questions only (mean ± std across 3 seeds)")
print("=" * 82)
print(f"{'Model':<18} {'Budget':<22} {'ECE±std':>16} {'Overconf':>10} {'Acc':>6}  Tag")
print("=" * 82)

summary_rows = []
for model in MODELS:
    for budget in BUDGETS:
        seed_eces, seed_overs, seed_accs = [], [], []
        for seed in range(N_SEEDS):
            sub = df_v[
                (df_v["model"]   == model["label"]) &
                (df_v["budget"]  == budget["label"]) &
                (df_v["is_trap"]) &
                (df_v["seed"]    == seed)
            ]
            if len(sub) < 5:
                continue
            ece, _ = binned_ece(sub["confidence"].tolist(), sub["correct"].tolist())
            seed_eces.append(ece)
            seed_overs.append(sub["confidence"].mean() - sub["correct"].mean())
            seed_accs.append(sub["correct"].mean())

        if not seed_eces:
            continue
        me  = np.mean(seed_eces)
        se  = np.std(seed_eces)
        mo  = np.mean(seed_overs)
        ma  = np.mean(seed_accs)
        tag = "OVERCONF" if mo > 0.05 else ("underconf" if mo < -0.05 else "calibrated")
        summary_rows.append({
            "model":         model["label"],
            "budget":        budget["label"],
            "mean_ece":      round(me, 4),
            "std_ece":       round(se, 4),
            "mean_overconf": round(mo, 3),
            "mean_acc":      round(ma, 3),
        })
        print(f"{model['label']:<18} {budget['label']:<22} "
              f"{me:.4f}±{se:.4f}  {mo:>+9.3f}  {ma:>6.3f}  {tag}")
    print()

# ================== STATISTICAL TEST ==================
print("\nSTATISTICAL TEST — 70B: medium vs heavy ECE (paired t-test)")
m_eces, h_eces = [], []
for seed in range(N_SEEDS):
    sub_m = df_v[
        (df_v["model"]   == "Llama-3.3-70B") &
        (df_v["budget"]  == "medium_reasoning") &
        (df_v["is_trap"]) &
        (df_v["seed"]    == seed)
    ]
    sub_h = df_v[
        (df_v["model"]   == "Llama-3.3-70B") &
        (df_v["budget"]  == "heavy_reasoning") &
        (df_v["is_trap"]) &
        (df_v["seed"]    == seed)
    ]
    if len(sub_m) > 0:
        ece_m, _ = binned_ece(sub_m["confidence"].tolist(), sub_m["correct"].tolist())
        m_eces.append(ece_m)
    if len(sub_h) > 0:
        ece_h, _ = binned_ece(sub_h["confidence"].tolist(), sub_h["correct"].tolist())
        h_eces.append(ece_h)

if len(m_eces) >= 3 and len(h_eces) >= 3:
    t_stat, p_val = stats.ttest_rel(m_eces, h_eces)
    print(f"  Medium ECE : {np.mean(m_eces):.4f} ± {np.std(m_eces):.4f}")
    print(f"  Heavy  ECE : {np.mean(h_eces):.4f} ± {np.std(h_eces):.4f}")
    print(f"  t={t_stat:.3f}  p={p_val:.4f}")
    sig = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "not significant — run more seeds"
    print(f"  {sig}")

# ================== SMOKING GUN ==================
print("\n\nSMOKING GUN — wrong + confidence ≥ 0.90 (trap questions only)")
print("-" * 90)
wc = df_v[
    (df_v["correct"]    == 0) &
    (df_v["confidence"] >= 0.90) &
    (df_v["is_trap"])
][["model","budget","trap_type","expected","predicted","confidence"]]\
  .sort_values("confidence", ascending=False)
print(wc.head(25).to_string(index=False))

# ================== PLOTS ==================
sdf = pd.DataFrame(summary_rows)
budget_order = ["no_reasoning","light_reasoning","medium_reasoning","heavy_reasoning"]
budget_short = ["none","light","medium","heavy"]
C = {"Llama-3.1-8B": "#5DCAA5", "Llama-3.3-70B": "#7F77DD"}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Reasoning Budget vs Calibration on Reasoning-Trap Questions\n"
    f"Llama-3.1-8B vs Llama-3.3-70B  |  {N_SEEDS} seeds × "
    f"{sum(1 for q in TRAP_QUESTIONS if q['trap_type'] != 'easy')} trap questions",
    fontsize=13, fontweight="bold"
)

x = np.arange(len(budget_order))
w = 0.35

# ── Plot 1: ECE with error bars ───────────────────────────────────────
ax = axes[0][0]
for mi, model in enumerate(MODELS):
    sub  = sdf[sdf["model"] == model["label"]]
    eces = [sub[sub["budget"]==b]["mean_ece"].values[0]
            if len(sub[sub["budget"]==b]) else 0 for b in budget_order]
    stds = [sub[sub["budget"]==b]["std_ece"].values[0]
            if len(sub[sub["budget"]==b]) else 0 for b in budget_order]
    offset = (mi - 0.5) * w
    ax.bar(x+offset, eces, w, color=C[model["label"]], alpha=0.85,
           label=model["label"])
    ax.errorbar(x+offset, eces, yerr=stds, fmt="none",
                color="black", capsize=4, linewidth=1.5)
ax.axhline(0.10, color="orange", linestyle="--", linewidth=1,
           label="0.10 threshold", alpha=0.7)
ax.set_title("ECE on Trap Questions\n(mean ± std, 3 seeds)  — KEY RESULT",
             fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(budget_short)
ax.set_ylabel("Expected Calibration Error")
ax.legend(fontsize=9)

# ── Plot 2: Overconfidence gap ────────────────────────────────────────
ax = axes[0][1]
for mi, model in enumerate(MODELS):
    sub   = sdf[sdf["model"] == model["label"]]
    overs = [sub[sub["budget"]==b]["mean_overconf"].values[0]
             if len(sub[sub["budget"]==b]) else 0 for b in budget_order]
    offset = (mi - 0.5) * w
    ax.bar(x+offset, overs, w, color=C[model["label"]], alpha=0.85,
           label=model["label"])
ax.axhline(0,     color="black",  linewidth=0.8)
ax.axhline(0.05,  color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axhline(-0.05, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_title("Overconfidence Gap (conf − acc)\nTrap Questions")
ax.set_xticks(x); ax.set_xticklabels(budget_short)
ax.set_ylabel("Conf − Acc")
ax.legend(fontsize=9)

# ── Plot 3: Accuracy ──────────────────────────────────────────────────
ax = axes[0][2]
for mi, model in enumerate(MODELS):
    sub  = sdf[sdf["model"] == model["label"]]
    accs = [sub[sub["budget"]==b]["mean_acc"].values[0]
            if len(sub[sub["budget"]==b]) else 0 for b in budget_order]
    offset = (mi - 0.5) * w
    ax.bar(x+offset, accs, w, color=C[model["label"]], alpha=0.85,
           label=model["label"])
ax.set_title("Accuracy by Reasoning Budget\nTrap Questions")
ax.set_xticks(x); ax.set_xticklabels(budget_short)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)

# ── Plot 4: Wrong + confident by trap type ────────────────────────────
ax = axes[1][0]
wc_all = df_v[
    (df_v["correct"]    == 0) &
    (df_v["confidence"] >= 0.80) &
    (df_v["is_trap"])
]
tc = wc_all["trap_type"].value_counts()
bar_colors = ["#E24B4A" if v > 10 else
              "#EF9F27" if v > 5  else "#5DCAA5" for v in tc.values]
ax.barh(range(len(tc)), tc.values, color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(tc)))
ax.set_yticklabels(tc.index, fontsize=9)
ax.axvline(10, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_title("Wrong + Confident (≥ 0.80)\nBy Trap Type — Both Models, All Budgets")
ax.set_xlabel("Count (across models, budgets, seeds)")

# ── Plot 5: Reliability diagram — 70B heavy (degraded condition) ──────
ax = axes[1][1]
sub70h = df_v[
    (df_v["model"]   == "Llama-3.3-70B") &
    (df_v["budget"]  == "heavy_reasoning") &
    (df_v["is_trap"])
]
if len(sub70h) > 10:
    _, bd   = binned_ece(sub70h["confidence"].tolist(),
                         sub70h["correct"].tolist(), n_bins=8)
    mids    = [b["mid"]  for b in bd if b["acc"] is not None]
    baccs   = [b["acc"]  for b in bd if b["acc"] is not None]
    bconfs  = [b["conf"] for b in bd if b["acc"] is not None]
    ns      = [b["n"]    for b in bd if b["acc"] is not None]
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Perfect calibration", alpha=0.6)
    ax.bar(mids, baccs, width=0.1, alpha=0.45, color="#7F77DD", label="Accuracy in bin")
    ax.scatter(bconfs, baccs, color="#E24B4A", zorder=5, s=70,
               label="(conf, acc) observed")
    for m, a, n in zip(mids, baccs, ns):
        ax.annotate(f"n={n}", (m, min(a+0.04, 1.08)),
                    fontsize=7, ha="center", color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
    ax.set_title("Reliability Diagram\n70B Heavy Reasoning — Overconfidence Visible")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

# ── Plot 6: Reliability diagram — 70B medium (best condition) ─────────
ax = axes[1][2]
sub70m = df_v[
    (df_v["model"]   == "Llama-3.3-70B") &
    (df_v["budget"]  == "medium_reasoning") &
    (df_v["is_trap"])
]
if len(sub70m) > 10:
    _, bd   = binned_ece(sub70m["confidence"].tolist(),
                         sub70m["correct"].tolist(), n_bins=8)
    mids    = [b["mid"]  for b in bd if b["acc"] is not None]
    baccs   = [b["acc"]  for b in bd if b["acc"] is not None]
    bconfs  = [b["conf"] for b in bd if b["acc"] is not None]
    ns      = [b["n"]    for b in bd if b["acc"] is not None]
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Perfect calibration", alpha=0.6)
    ax.bar(mids, baccs, width=0.1, alpha=0.45, color="#5DCAA5", label="Accuracy in bin")
    ax.scatter(bconfs, baccs, color="#E24B4A", zorder=5, s=70,
               label="(conf, acc) observed")
    for m, a, n in zip(mids, baccs, ns):
        ax.annotate(f"n={n}", (m, min(a+0.04, 1.08)),
                    fontsize=7, ha="center", color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
    ax.set_title("Reliability Diagram\n70B Medium Reasoning — Best Calibration")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("paper_figure_1.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved → paper_figure_1.png")
print("Data  saved → full_results_v2.csv")
