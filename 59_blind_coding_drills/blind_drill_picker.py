"""
Pick a random blind coding drill.
"""
import random


DRILLS = [
    ("Stable Softmax", 15),
    ("Binary Logistic Regression Step", 15),
    ("Causal Mask", 15),
    ("Top-k Filtering", 15),
    ("Masked Softmax", 20),
    ("K-Means One Iteration", 20),
    ("Decision Tree Best Split", 20),
    ("Pairwise Squared Distances", 20),
    ("Attention From Scratch", 30),
    ("Beam Search Skeleton", 30),
    ("Bootstrap Confidence Interval", 30),
    ("Data Leakage Check", 30),
]


if __name__ == "__main__":
    name, minutes = random.choice(DRILLS)
    print("Blind Coding Drill Picker")
    print("=" * 60)
    print(f"Selected drill: {name}")
    print(f"Suggested timer: {minutes} minutes")
    print("Open drills.md only after you finish or to verify the exact prompt.")
