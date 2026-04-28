"""
Simple helper for mock interview practice.
Prints a random interview loop and suggested time box.
"""
import random


LOOPS = [
    ("Theory + Follow-Ups", 8),
    ("Probability / Statistics", 8),
    ("Coding", 20),
    ("Debugging", 10),
    ("Research Judgment", 10),
    ("Large-Scale Systems", 10),
    ("Paper Critique", 8),
    ("End-to-End Mixed Loop", 15),
]


if __name__ == "__main__":
    loop_name, minutes = random.choice(LOOPS)
    print("Mock Interview Picker")
    print("=" * 60)
    print(f"Selected loop: {loop_name}")
    print(f"Suggested time box: {minutes} minutes")
    print("Open mock_loops.md and run that section without notes.")
