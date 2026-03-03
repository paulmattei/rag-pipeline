import json

import matplotlib.pyplot as plt

with open("cache/chunks.json", "r") as f:
    chunks = json.load(f)

raw_sizes = [len(c) for c in chunks]
stripped_sizes = [len(c.strip()) for c in chunks]
whitespace_sizes = [r - s for r, s in zip(raw_sizes, stripped_sizes)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(raw_sizes, bins=100)
axes[0, 0].set_title(f"Raw char length (n={len(chunks)})")
axes[0, 0].set_xlabel("chars")

axes[0, 1].hist(stripped_sizes, bins=100)
axes[0, 1].set_title("Stripped char length")
axes[0, 1].set_xlabel("chars")

axes[1, 0].hist(whitespace_sizes, bins=100)
axes[1, 0].set_title("Leading/trailing whitespace")
axes[1, 0].set_xlabel("chars")

short = [c for c in chunks if len(c.strip()) < 100]
short_sizes = [len(c.strip()) for c in short]
axes[1, 1].hist(short_sizes, bins=50)
axes[1, 1].set_title(f"Short chunks (<100 stripped chars, n={len(short)})")
axes[1, 1].set_xlabel("chars")

plt.tight_layout()
plt.show()

print(f"Total chunks: {len(chunks)}")
print(f"Chunks < 100 stripped chars: {len(short)} ({100*len(short)/len(chunks):.1f}%)")
print(f"Chunks < 50 stripped chars: {sum(1 for c in chunks if len(c.strip()) < 50)}")
print("\nSample short chunks:")
for c in short[:5]:
    print(f"  [{len(c.strip()):3d} chars] {c.strip()[:80]!r}")
