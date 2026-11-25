import sys, json
from dotenv import load_dotenv
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()

judge = AnthropicModel(model="claude-sonnet-4-6")

faithfulness = FaithfulnessMetric(model=judge)
correctness = GEval(
    name="Correctness",
    criteria="Is the actual output actually consistent with the expected output? Ignore extra detail and citation markers like [1].",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    model=judge,
)

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    results = json.load(f)

scores = []
for r in results:
    if not r["answer"].strip():
        scores.append({"question": r["question"], "faithfulness": 0.0, "correctness": 0.0})
        continue
    
    case = LLMTestCase(
        input=r["question"],
        actual_output=r["answer"],
        expected_output=r["expected_answer"],
        retrieval_context=r["contexts"],
    )
    faithfulness.measure(case)
    correctness.measure(case)
    scores.append({
        "question": r["question"],
        "faithfulness": faithfulness.score,
        "correctness": correctness.score,
    })

n = len(scores)
f_mean = sum(s["faithfulness"] for s in scores) / n
c_mean = sum(s["correctness"] for s in scores) / n
print(f"{path} Faithfulness={f_mean:.3f} Correctness={c_mean:.3f}")

with open(path.replace("results", "scores"), "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2)