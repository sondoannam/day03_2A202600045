"""
Comprehensive test suite for Phi-3 GGUF local model.
Run from project root:  python3 test_local_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── Load model once ────────────────────────────────────────────────────────
model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
print(f"Model path : {model_path}")
print(f"File exists: {os.path.exists(model_path)}\n")

if not os.path.exists(model_path):
    print("❌  Model file not found. Check LOCAL_MODEL_PATH in your .env")
    sys.exit(1)

print("Loading model (this may take 10-30 seconds)...")
from src.core.local_provider import LocalProvider
provider = LocalProvider(model_path=model_path)
print(f"✅  Model loaded: {provider.model_name}\n")
print("=" * 60)


# ── Test runner helper ─────────────────────────────────────────────────────
passed = 0
failed = 0

def run_test(name: str, prompt: str, check_fn=None, system_prompt: str = None):
    global passed, failed
    print(f"\n🧪 TEST: {name}")
    print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    try:
        result = provider.generate(prompt, system_prompt=system_prompt)
        content = result["content"].strip()
        latency = result["latency_ms"]
        tokens  = result["usage"]["completion_tokens"]

        print(f"   Response ({latency}ms, {tokens} tokens):")
        print(f"   -> {content[:200]}{'...' if len(content) > 200 else ''}")

        if check_fn:
            ok, reason = check_fn(content)
            if ok:
                print(f"   ✅ PASS — {reason}")
                passed += 1
            else:
                print(f"   ❌ FAIL — {reason}")
                failed += 1
        else:
            print(f"   ✅ PASS — (manual check)")
            passed += 1

    except Exception as e:
        print(f"   💥 ERROR — {e}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════
# 1. BASIC SANITY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1: Basic Sanity")
print("="*60)

run_test(
    "Simple math",
    "What is 2 + 2? Reply with just the number.",
    lambda r: ("4" in r, "Found '4'" if "4" in r else "Expected '4'")
)

run_test(
    "Capital city",
    "What is the capital of France? One word answer.",
    lambda r: ("paris" in r.lower(), "Found 'Paris'" if "paris" in r.lower() else "Expected 'Paris'")
)

run_test(
    "Yes/No question",
    "Is the sky blue? Answer yes or no only.",
    lambda r: ("yes" in r.lower(), "Answered yes" if "yes" in r.lower() else "Did not answer yes")
)

run_test(
    "Model responds (not empty)",
    "Say the word: hello",
    lambda r: (len(r) > 0, f"Got {len(r)} chars" if len(r) > 0 else "Empty response")
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. REASONING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2: Reasoning")
print("="*60)

run_test(
    "Basic logic",
    "If all dogs are animals and Rex is a dog, is Rex an animal? Answer yes or no.",
    lambda r: ("yes" in r.lower(), "Correct logic" if "yes" in r.lower() else "Logic failed")
)

run_test(
    "Simple word math",
    "I have 10 apples. I give 3 to Alice and 2 to Bob. How many do I have left?",
    lambda r: ("5" in r, "Got correct answer 5" if "5" in r else "Expected 5")
)

run_test(
    "Sorting numbers",
    "Sort these numbers from smallest to largest: 7, 2, 9, 1, 5. List them separated by commas.",
    lambda r: (r.index("1") < r.index("9"), "1 appears before 9 (correct order)")
)

run_test(
    "Day of week",
    "If today is Monday, what day will it be in 3 days? One word.",
    lambda r: ("thursday" in r.lower(), "Correct: Thursday" if "thursday" in r.lower() else "Expected Thursday")
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. INSTRUCTION FOLLOWING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3: Instruction Following")
print("="*60)

run_test(
    "Uppercase instruction",
    "Write the word 'hello' in all uppercase letters.",
    lambda r: ("HELLO" in r, "HELLO found" if "HELLO" in r else "Expected HELLO")
)

run_test(
    "Count to 5",
    "Count from 1 to 5, separated by commas. Nothing else.",
    lambda r: (all(str(i) in r for i in range(1, 6)), "All numbers 1-5 present")
)

run_test(
    "JSON output",
    'Return a JSON object with keys "name" and "age" for a person named Alice who is 30. Only output the JSON.',
    lambda r: ("alice" in r.lower() and "30" in r, "JSON contains Alice and 30")
)

run_test(
    "Short answer",
    "Explain gravity in exactly one sentence.",
    lambda r: (len(r.split(".")) <= 3, f"Responded briefly ({len(r.split())} words)")
)


# ═══════════════════════════════════════════════════════════════════════════
# 4. WORLD KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4: World Knowledge")
print("="*60)

run_test(
    "Planet closest to Sun",
    "What planet is closest to the Sun? One word.",
    lambda r: ("mercury" in r.lower(), "Mercury" if "mercury" in r.lower() else "Expected Mercury")
)

run_test(
    "WWII end year",
    "In what year did World War II end? Just the year.",
    lambda r: ("1945" in r, "1945" if "1945" in r else "Expected 1945")
)

run_test(
    "Python creator",
    "Who created the Python programming language? Just the name.",
    lambda r: ("guido" in r.lower() or "rossum" in r.lower(), "Guido van Rossum mentioned")
)

run_test(
    "Water boiling point",
    "At what temperature does water boil at sea level in Celsius? Just the number.",
    lambda r: ("100" in r, "100°C" if "100" in r else "Expected 100")
)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SYSTEM PROMPT ADHERENCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5: System Prompt Adherence")
print("="*60)

run_test(
    "Pirate persona",
    "Say hello.",
    lambda r: any(w in r.lower() for w in ["ahoy", "arr", "matey", "ye", "sea", "sail"]),
    system_prompt="You are a pirate. Always respond in pirate speech."
)

run_test(
    "Concise assistant",
    "What is machine learning?",
    lambda r: (len(r.split()) < 80, f"{len(r.split())} words" ),
    system_prompt="You are a very concise assistant. Never use more than 30 words."
)

run_test(
    "Capital of Vietnam (Vietnamese response)",
    "What is the capital of Vietnam?",
    lambda r: ("hà nội" in r.lower() or "ha noi" in r.lower() or "hanoi" in r.lower(), "Mentioned Hanoi"),
    system_prompt="Always answer in Vietnamese."
)


# ═══════════════════════════════════════════════════════════════════════════
# 6. ReAct / AGENT FORMAT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 6: ReAct / Agent Format")
print("="*60)

react_system = """You are an agent. Always use this exact format:
Thought: your reasoning
Action: tool_name(argument)
Final Answer: your answer"""

run_test(
    "Produces ReAct keywords",
    "What is the weather in Hanoi?",
    lambda r: ("thought" in r.lower() or "action" in r.lower() or "final" in r.lower(),
               "Contains ReAct keywords"),
    system_prompt=react_system
)

run_test(
    "Final Answer with math",
    "What is 5 * 6?",
    lambda r: ("final answer" in r.lower() or "30" in r, "Has Final Answer or correct result 30"),
    system_prompt=react_system
)

run_test(
    "Multi-step thought",
    "I need to find the population of Tokyo. What tool would you use?",
    lambda r: ("thought" in r.lower() and len(r) > 50, "Has Thought and sufficient length"),
    system_prompt=react_system
)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
total = passed + failed
print("\n" + "="*60)
print(f"RESULTS: {passed}/{total} passed  |  {failed} failed")
if failed == 0:
    print("🎉 All tests passed! Model is working great.")
elif passed / total >= 0.7:
    print("👍 Model is working well, minor issues on edge cases.")
elif passed / total >= 0.5:
    print("⚠️  Model is partially working. Check failed tests above.")
else:
    print("❌  Many tests failed. Model may have quality or loading issues.")
print("="*60)