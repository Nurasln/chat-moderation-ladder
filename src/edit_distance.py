"""Classical NLP step 2 - minimum edit distance (Levenshtein), written by hand.

The edit distance between two words is the smallest number of single-letter
insertions, deletions or substitutions that turn one into the other:
    "idiot"  -> "idiiot"  : 1 (one insertion)
    "stupid" -> "stupld"  : 1 (one substitution)

We fill a table D where D[i][j] = distance between the first i letters of `a`
and the first j letters of `b` (dynamic programming, as in the course slides).
"""


def edit_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Levenshtein distance. If max_dist is given we stop early once it is clearly exceeded."""
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1  # lengths differ too much, no need to compute

    prev = list(range(len(b) + 1))           # row for i = 0: "" -> b[:j] costs j insertions
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)              # column j = 0: a[:i] -> "" costs i deletions
        for j in range(1, len(b) + 1):
            sub_cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,                  # delete a[i-1]
                cur[j - 1] + 1,               # insert b[j-1]
                prev[j - 1] + sub_cost,       # substitute (or keep if equal)
            )
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1               # every path is already too expensive
        prev = cur                            # we only ever need the previous row
    return prev[-1]


if __name__ == "__main__":
    for a, b in [("idiot", "idiiot"), ("stupid", "stupld"), ("kitten", "sitting"), ("hate", "love")]:
        print(f"{a:7} vs {b:8} -> {edit_distance(a, b)}")
