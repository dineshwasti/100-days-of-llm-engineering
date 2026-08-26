# 100 numpy exercises — my solutions

My worked solutions to the classic **100 numpy exercises** set.

> **Source & credit:** the exercise questions come from
> [rougier/numpy-100](https://github.com/rougier/numpy-100) by Nicolas P. Rougier (MIT licensed).
> The questions are his; the solutions and notes here are mine.

## Why I did this

Numpy is the substrate under almost everything in ML and LLM work — embeddings are
just float arrays, attention is just matmuls, and every vector store is doing
cosine similarity on numpy-shaped data. Working through these got me comfortable
with broadcasting, views vs copies, structured dtypes, and the `ufunc` tricks
(`np.add.at`, `reduceat`, `einsum`) that turn slow Python loops into one-liners.

## Files

| File | What's in it |
|------|--------------|
| [solutions.md](./solutions.md) | All 100 questions with my answers and short notes |

## Notes on how I worked through these

- I ran everything against numpy 2.x. A few of the original answers rely on APIs
  that have since moved (`np.core.records`, `stride_tricks` gymnastics), so where
  there's a modern equivalent I used it — `np.lib.stride_tricks.sliding_window_view`
  in particular replaces a lot of the old manual-stride code.
- The "what is the output of this expression" questions (17, 26, 27, 28) were the
  most useful ones. They're not about numpy tricks, they're about knowing where
  numpy quietly disagrees with plain Python.
- Anything marked ★★★ I'd want to look up again rather than claim I'd write from
  memory. The point was understanding the approach, not memorizing it.
