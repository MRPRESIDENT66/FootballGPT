"""Evaluate Router intent classification accuracy.

Runs a set of test queries through the Router and checks if the predicted
intent matches the expected intent. Reports accuracy per intent and overall.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.router import route_query

TEST_CASES = [
    # Scout intent
    {"query": "Find me a fast right winger under 23", "expected": "scout"},
    {"query": "Who are the best young strikers in La Liga?", "expected": "scout"},
    {"query": "Show me defenders with the most tackles", "expected": "scout"},
    {"query": "I need a goalkeeper from Serie A", "expected": "scout"},
    {"query": "Search for midfielders with high pass accuracy", "expected": "scout"},

    # Compare intent
    {"query": "Compare Salah and Haaland", "expected": "compare"},
    {"query": "Who is better, Vinicius or Mbappe?", "expected": "compare"},
    {"query": "Salah vs Son stats comparison", "expected": "compare"},
    {"query": "How does Bruno Fernandes compare to De Bruyne?", "expected": "compare"},
    {"query": "Between Rodri and Rice, who is the better CDM?", "expected": "compare"},

    # Tactics intent
    {"query": "Analyze Barcelona's tactical system", "expected": "tactics"},
    {"query": "What is Liverpool's playing style?", "expected": "tactics"},
    {"query": "How does Guardiola set up Man City?", "expected": "tactics"},
    {"query": "Explain Arsenal's defensive weaknesses", "expected": "tactics"},
    {"query": "What formation does Bayern Munich use?", "expected": "tactics"},

    # Recommend intent
    {"query": "Recommend a striker for Arsenal", "expected": "recommend"},
    {"query": "Who should Liverpool sign to replace Salah?", "expected": "recommend"},
    {"query": "Find a right-back that fits Real Madrid's system", "expected": "recommend"},
    {"query": "Suggest a midfielder for Chelsea under 25", "expected": "recommend"},
    {"query": "What player should Barcelona buy for left-back?", "expected": "recommend"},
]


def main():
    print("⚽ Router Intent Classification Evaluation")
    print(f"📊 Test cases: {len(TEST_CASES)}")
    print("=" * 60)

    results = {"correct": 0, "wrong": 0, "error": 0}
    per_intent = {}

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected = case["expected"]

        if expected not in per_intent:
            per_intent[expected] = {"correct": 0, "total": 0}
        per_intent[expected]["total"] += 1

        try:
            result = route_query(query)
            predicted = result.get("intent", "unknown")
            match = predicted == expected
            icon = "✅" if match else "❌"

            print(f"  [{i:2d}] {icon} \"{query[:50]}...\"")
            if not match:
                print(f"       Expected: {expected} | Got: {predicted}")

            if match:
                results["correct"] += 1
                per_intent[expected]["correct"] += 1
            else:
                results["wrong"] += 1

        except Exception as e:
            print(f"  [{i:2d}] ⚠️  Error: {e}")
            results["error"] += 1

    # Summary
    total = len(TEST_CASES)
    accuracy = results["correct"] / total * 100

    print("\n" + "=" * 60)
    print(f"📊 Overall Accuracy: {results['correct']}/{total} = {accuracy:.1f}%")
    print()

    print("Per-intent breakdown:")
    for intent, stats in per_intent.items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"  {intent:12s}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

    if results["error"] > 0:
        print(f"\n⚠️  Errors: {results['error']}")

    print()
    if accuracy >= 90:
        print("✅ Router classification accuracy is strong.")
    elif accuracy >= 75:
        print("⚠️  Router accuracy needs improvement.")
    else:
        print("❌ Router accuracy is below acceptable threshold.")


if __name__ == "__main__":
    main()
