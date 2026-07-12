# Answer Key — "Your Turn: Calculate Similarity Yourself"

Worked solutions for the three practice problems on the slide. Students should
arrive at the same cosine values (small rounding is fine). All vectors use
**integers** so the dot products are simple paper/calculator arithmetic.

Formula: **cos(A, B) = (A · B) / (|A| × |B|)**

> Note: the embeddings are scaled by 10 (e.g. [6, 9, 1] instead of [0.6, 0.9, 0.1]).
> Cosine similarity is scale-invariant, so the final cos values are identical.

---

## Problem 1 — dog vs tree

- dog = [6, 9, 1]
- tree = [-7, -8, -3]

**Dot product:**
```
dog · tree = (6 × -7) + (9 × -8) + (1 × -3)
           = -42 + (-72) + (-3)
           = -117
```

**Lengths:**
```
|dog|  = √(6² + 9² + 1²) = √(36 + 81 + 1) = √118  ≈ 10.86
|tree| = √((-7)² + (-8)² + (-3)²) = √(49 + 64 + 9) = √122 ≈ 11.05
```

**Cosine similarity:**
```
cos = -117 / (10.86 × 11.05)
    = -117 / 120.00
    ≈ -0.98
```

**Interpretation:** Negative and close to -1 → dog and tree point in **opposite
directions** (very different meaning). ✅

---

## Problem 2 — dog vs man

- dog = [6, 9, 1]
- man = [6, -2, 8]

**Dot product:**
```
dog · man = (6 × 6) + (9 × -2) + (1 × 8)
          = 36 + (-18) + 8
          = 26
```

**Lengths:**
```
|dog| = √118 ≈ 10.86   (from Problem 1)
|man| = √(6² + (-2)² + 8²) = √(36 + 4 + 64) = √104 ≈ 10.20
```

**Cosine similarity:**
```
cos = 26 / (10.86 × 10.20)
    = 26 / 110.77
    ≈ 0.23
```

**Interpretation:** Small positive value → dog and man are only **weakly related**
(they share some direction but are far from identical). ✅

---

## Problem 3 — man vs woman

- man = [6, -2, 8]
- woman = [7, 3, 9]

**Dot product:**
```
man · woman = (6 × 7) + (-2 × 3) + (8 × 9)
            = 42 + (-6) + 72
            = 108
```

**Lengths:**
```
|man|   = √104 ≈ 10.20   (from Problem 2)
|woman| = √(7² + 3² + 9²) = √(49 + 9 + 81) = √139 ≈ 11.79
```

**Cosine similarity:**
```
cos = 108 / (10.20 × 11.79)
    = 108 / 120.26
    ≈ 0.90
```

**Interpretation:** High positive value → man and woman are **very similar** (both
human). This matches the matrix on the previous slide. ✅

---

## Summary table (for quick checking)

| Pair | Dot product | |A| | |B| | Cosine | Meaning |
|------|------------|-----|-----|--------|---------|
| dog – tree | -117 | √118 ≈ 10.86 | √122 ≈ 11.05 | **-0.98** | Opposite |
| dog – man | 26 | √118 ≈ 10.86 | √104 ≈ 10.20 | **0.23** | Weakly related |
| man – woman | 108 | √104 ≈ 10.20 | √139 ≈ 11.79 | **0.90** | Very similar |

These match the cosine-similarity matrix shown on the "How Meaning Works" slide.
