# 100 numpy exercises — solutions

Questions from [rougier/numpy-100](https://github.com/rougier/numpy-100) (MIT, Nicolas P. Rougier).
Solutions and notes are my own. Tested against numpy 2.x.

---

#### 1. Import the numpy package under the name `np` (★☆☆)

```python
import numpy as np
```

#### 2. Print the numpy version and the configuration (★☆☆)

```python
print(np.__version__)
np.show_config()
```

#### 3. Create a null vector of size 10 (★☆☆)

```python
Z = np.zeros(10)
```

#### 4. How to find the memory size of any array (★☆☆)

```python
Z = np.zeros((10, 10))
print(Z.nbytes)                  # 800
print(Z.size * Z.itemsize)       # same thing, the long way
```

`nbytes` is just `size * itemsize`. Worth knowing both — the second one makes it
obvious *why* a float64 array is 8x a int8 array of the same shape.

#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)

```bash
python -c "import numpy; numpy.info(numpy.add)"
```

#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

```python
Z = np.zeros(10)
Z[4] = 1
```

Index 4, not 5 — "fifth value" is zero-indexed here.

#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)

```python
Z = np.arange(10, 50)
```

#### 8. Reverse a vector (first element becomes last) (★☆☆)

```python
Z = np.arange(50)
Z = Z[::-1]
```

Note this gives you a *view*, not a copy — writing to it writes to the original.

#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

```python
Z = np.arange(9).reshape(3, 3)
```

#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)

```python
nz = np.nonzero([1, 2, 0, 0, 4, 0])
print(nz)     # (array([0, 1, 4]),)
```

It returns a *tuple* of index arrays, one per dimension — which is why there's a
trailing comma for a 1D input.

#### 11. Create a 3x3 identity matrix (★☆☆)

```python
Z = np.eye(3)
```

#### 12. Create a 3x3x3 array with random values (★☆☆)

```python
rng = np.random.default_rng()
Z = rng.random((3, 3, 3))
```

I'm using the modern `default_rng()` generator rather than `np.random.random` —
it's the recommended API now and makes seeding explicit.

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

```python
Z = rng.random((10, 10))
print(Z.min(), Z.max())
```

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)

```python
Z = rng.random(30)
print(Z.mean())
```

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

```python
Z = np.ones((10, 10))
Z[1:-1, 1:-1] = 0
```

Fill it solid, then hollow out the middle with slicing — easier than building the
border piece by piece.

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

```python
Z = np.ones((5, 5))
Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0)
```

#### 17. What is the result of the following expression? (★☆☆)

```python
0 * np.nan            # nan   -- nan poisons any arithmetic it touches
np.nan == np.nan      # False -- nan is not equal to itself, by IEEE-754
np.inf > np.nan       # False -- any comparison with nan is False
np.nan - np.nan       # nan
np.nan in set([np.nan])  # True  -- surprising! sets check identity first,
                         #          and it's literally the same object
0.3 == 3 * 0.1        # False -- classic float rounding, 3*0.1 is 0.30000000000000004
```

The `np.nan in set(...)` one is the good trick question: `==` says no, but `in`
says yes, because CPython short-circuits on `is` before falling back to `==`.

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

```python
Z = np.diag(1 + np.arange(4), k=-1)
```

`k=-1` shifts the diagonal down one.

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

```python
Z = np.zeros((8, 8), dtype=int)
Z[1::2, ::2] = 1
Z[::2, 1::2] = 1
```

Two strided assignments: odd rows/even cols, then even rows/odd cols.

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

```python
print(np.unravel_index(99, (6, 7, 8)))   # (1, 5, 3)
```

99, not 100 — the "100th element" is at flat index 99.

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

```python
Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
```

Much nicer than #19: define the 2x2 unit, repeat it.

#### 22. Normalize a 5x5 random matrix (★☆☆)

```python
Z = rng.random((5, 5))
Z = (Z - Z.mean()) / Z.std()
```

This is z-score normalization (mean 0, std 1). If you wanted min-max scaling to
[0,1] instead it'd be `(Z - Z.min()) / (Z.max() - Z.min())` — worth being explicit
about which one you mean.

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

```python
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])
```

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

```python
Z = np.ones((5, 3)) @ np.ones((3, 2))
```

`@` is the matrix product. `*` would be elementwise and would fail here.

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

```python
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
```

Note `&`, not `and` — Python's `and` can't operate elementwise on arrays.

#### 26. What is the output of the following script? (★☆☆)

```python
print(sum(range(5), -1))        # 9
from numpy import *
print(sum(range(5), -1))        # 10
```

Same line, different answers. Builtin `sum(iterable, start)` treats `-1` as a
starting value: `0+1+2+3+4 + (-1) = 9`. After the star-import, `sum` is now
`np.sum(a, axis)`, so `-1` means "last axis" and you get `10`.

Good argument against `from numpy import *`.

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

```python
Z**Z          # legal
2 << Z >> 2   # legal for ints (bit shifts)
Z <- Z        # legal -- it's Z < (-Z), not an assignment arrow
1j*Z          # legal, promotes to complex
Z/1/1         # legal
Z<Z>Z         # ILLEGAL -- ValueError
```

The last one is chained comparison, which Python expands to
`(Z<Z) and (Z>Z)` — and `and` needs a single truth value from an array, so it
raises "truth value of an array is ambiguous".

`Z <- Z` is the sneaky one. It looks like an assignment arrow from R; it's just
a comparison against the negation.

#### 28. What are the result of the following expressions? (★☆☆)

```python
np.array(0) / np.array(0)                       # nan  (+ RuntimeWarning)
np.array(0) // np.array(0)                      # 0    (+ RuntimeWarning)
np.array([np.nan]).astype(int).astype(float)    # some garbage value
```

Integer division by zero doesn't raise like plain Python does — you get a warning
and a sentinel. And casting `nan` to int is undefined behaviour: on most platforms
you get `-9.22e+18` (the int64 minimum). Never rely on that last one.

#### 29. How to round away from zero a float array ? (★☆☆)

```python
Z = rng.uniform(-10, 10, 10)
print(np.copysign(np.ceil(np.abs(Z)), Z))
```

Take the magnitude, round it up, then put the original sign back. Plain
`np.round` uses banker's rounding and rounds toward even, which isn't the same.

#### 30. How to find common values between two arrays? (★☆☆)

```python
print(np.intersect1d(Z1, Z2))
```

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

```python
with np.errstate(all="ignore"):
    np.arange(3) / 0
```

The context manager is much safer than the global `np.seterr(all="ignore")`,
because it puts the settings back when the block exits.

#### 32. Is the following expressions true? (★☆☆)

```python
np.sqrt(-1) == np.emath.sqrt(-1)   # False
```

`np.sqrt(-1)` gives `nan` (staying in the reals), while `np.emath.sqrt(-1)`
gives `1j` (promoting to complex). And even if both were nan, nan != nan.

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

```python
today     = np.datetime64("today", "D")
yesterday = today - np.timedelta64(1, "D")
tomorrow  = today + np.timedelta64(1, "D")
```

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

```python
Z = np.arange("2016-07", "2016-08", dtype="datetime64[D]")
```

`arange` works on datetimes — the `[D]` unit means it steps by days.

#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

```python
A = np.ones(3) * 1
B = np.ones(3) * 2

np.add(A, B, out=B)        # B = A + B
np.divide(A, 2, out=A)     # A = A / 2
np.negative(A, out=A)      # A = -A/2
np.multiply(A, B, out=A)   # A = (-A/2) * (A+B)
```

Every ufunc takes `out=`, which is how you avoid allocating temporaries. This
matters when arrays get big — it's the difference between one allocation and four.

#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)

```python
Z = rng.uniform(0, 10, 10)

print(Z - Z % 1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))
```

These agree for positive numbers only. For negatives, `floor(-2.5)` is `-3` but
`trunc(-2.5)` is `-2` — hence the question saying "positive".

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

```python
Z = np.zeros((5, 5))
Z += np.arange(5)
```

Broadcasting: the shape-(5,) row is stretched across all 5 rows.

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

```python
def generate():
    yield from range(10)

Z = np.fromiter(generate(), dtype=float, count=-1)
```

`count=-1` means "read until exhausted". Passing the real count is faster since
numpy can preallocate.

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

```python
Z = np.linspace(0, 1, 11, endpoint=False)[1:]
```

Ask for 11 points without the endpoint, then drop the leading 0. That leaves 10
values, both ends excluded.

#### 40. Create a random vector of size 10 and sort it (★★☆)

```python
Z = rng.random(10)
Z.sort()
```

`Z.sort()` is in place; `np.sort(Z)` returns a new array.

#### 41. How to sum a small array faster than np.sum? (★★☆)

```python
Z = np.arange(10)
np.add.reduce(Z)
```

`np.sum` dispatches through some Python-level machinery first; `np.add.reduce` is
the underlying ufunc call. Only measurably faster for *small* arrays where that
overhead dominates — for big arrays they're identical.

#### 42. Consider two random arrays A and B, check if they are equal (★★☆)

```python
np.array_equal(A, B)     # exact: shapes and all values identical
np.allclose(A, B)        # tolerant: fine for floats after arithmetic
```

For anything that's been through floating-point math, use `allclose`.

#### 43. Make an array immutable (read-only) (★★☆)

```python
Z = np.zeros(10)
Z.flags.writeable = False
```

Now `Z[0] = 1` raises a ValueError.

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

```python
Z = rng.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.hypot(X, Y)
T = np.arctan2(Y, X)
```

Use `arctan2(y, x)`, not `arctan(y/x)` — it gets the quadrant right and doesn't
blow up when x is 0. `hypot` avoids overflow compared to `sqrt(x**2 + y**2)`.

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

```python
Z = rng.random(10)
Z[Z.argmax()] = 0
```

#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)

```python
Z = np.zeros((5, 5), dtype=[("x", float), ("y", float)])
Z["x"], Z["y"] = np.meshgrid(np.linspace(0, 1, 5),
                             np.linspace(0, 1, 5))
```

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

```python
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
```

`np.subtract.outer` gives every pairwise difference in one shot — this is the
outer-product pattern generalized to any ufunc.

#### 48. Print the minimum and maximum representable values for each numpy scalar type (★★☆)

```python
for dtype in [np.int8, np.int32, np.int64]:
    print(dtype, np.iinfo(dtype).min, np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    info = np.finfo(dtype)
    print(dtype, info.min, info.max, info.eps)
```

`iinfo` for integers, `finfo` for floats. `finfo(...).eps` is the one to remember
— it's the smallest difference the type can actually represent near 1.0.

#### 49. How to print all the values of an array? (★★☆)

```python
with np.printoptions(threshold=sys.maxsize):
    print(np.zeros((40, 40)))
```

By default numpy truncates big arrays with `...`. The context manager form means
you don't permanently change print behaviour for the rest of the program.

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

```python
Z = np.arange(100)
v = rng.uniform(0, 100)
print(Z[np.abs(Z - v).argmin()])
```

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

```python
Z = np.zeros(10, dtype=[("position", [("x", float), ("y", float)]),
                        ("color",    [("r", float), ("g", float), ("b", float)])])
print(Z["position"]["x"])
```

Nested dtypes — you can compose them as deeply as you like.

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

```python
Z = rng.random((100, 2))
diff = Z[:, None, :] - Z[None, :, :]      # (100, 100, 2)
D = np.sqrt((diff ** 2).sum(axis=-1))     # (100, 100)
```

`scipy.spatial.distance.cdist(Z, Z)` does the same thing and is faster, but the
broadcasting version is worth writing once to see how the `None` axis-insertion
trick works. This exact pattern shows up constantly in embedding similarity code.

#### 53. How to convert a float (32 bits) array into an integer (32 bits) array in place?

```python
Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
```

Truly in place isn't really possible across dtypes — `copy=False` just means
"don't copy if you don't have to". The old `Z.view(np.int32)` trick reinterprets
the raw bits and gives you nonsense numbers, so don't use it here.

#### 54. How to read the following file? (★★☆)

```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

```python
from io import StringIO

s = StringIO("""1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11""")
Z = np.genfromtxt(s, delimiter=",", dtype=int)
```

`genfromtxt` (not `loadtxt`) is the one that copes with the missing fields —
they come back as `-1` for int dtype, or `nan` if you read as float.

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

```python
Z = np.arange(9).reshape(3, 3)

for index, value in np.ndenumerate(Z):
    print(index, value)

for index in np.ndindex(Z.shape):
    print(index, Z[index])
```

`ndenumerate` gives you index+value, `ndindex` gives just the indices.

#### 56. Generate a generic 2D Gaussian-like array (★★☆)

```python
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
D = np.hypot(X, Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
```

#### 57. How to randomly place p elements in a 2D array? (★★☆)

```python
n, p = 10, 3
Z = np.zeros((n, n))
np.put(Z, rng.choice(n * n, p, replace=False), 1)
```

`replace=False` matters — otherwise you can place fewer than p elements because
of duplicate indices.

#### 58. Subtract the mean of each row of a matrix (★★☆)

```python
X = rng.random((5, 10))
Y = X - X.mean(axis=1, keepdims=True)
```

`keepdims=True` is the important bit: it keeps the result shape (5,1) so it
broadcasts back against (5,10). Without it you get shape (5,) and it broadcasts
along the wrong axis.

#### 59. How to sort an array by the nth column? (★★☆)

```python
Z = rng.integers(0, 10, (3, 3))
n = 1
print(Z[Z[:, n].argsort()])
```

`argsort` gives the row order; using it as an index reorders whole rows.

#### 60. How to tell if a given 2D array has null columns? (★★☆)

```python
Z = rng.integers(0, 3, (3, 10))
print((~Z.any(axis=0)).any())
```

`Z.any(axis=0)` is per-column "has any nonzero"; negate it to find all-zero
columns, then ask if any exist.

#### 61. Find the nearest value from a given value in an array (★★☆)

```python
Z = rng.uniform(0, 1, 10)
z = 0.5
print(Z.flat[np.abs(Z - z).argmin()])
```

`.flat` makes this work for any dimensionality, since `argmin` returns a flat index.

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

```python
A = np.arange(3).reshape(1, 3)
B = np.arange(3).reshape(3, 1)
it = np.nditer([A, B, None])
for x, y, z in it:
    z[...] = x + y
print(it.operands[2])
```

Passing `None` as the third operand lets nditer allocate the output with the
correct broadcast shape (3,3). In practice you'd just write `A + B`.

#### 63. Create an array class that has a name attribute (★★☆)

```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", "no name")

Z = NamedArray(np.arange(10), "range_10")
print(Z.name)
```

Subclassing ndarray needs both `__new__` and `__array_finalize__` — the latter is
what preserves your attribute through slicing and views.

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

```python
Z = np.ones(10)
I = rng.integers(0, len(Z), 20)
np.add.at(Z, I, 1)
```

The trap: `Z[I] += 1` does **not** work when `I` has duplicates. It reads, adds,
and writes back once, so repeats get counted a single time. `np.add.at` is the
unbuffered version that accumulates properly.

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

```python
X = [1, 2, 3, 4, 5, 6]
I = [1, 3, 9, 3, 4, 1]
F = np.bincount(I, weights=X)
```

Same problem as #64, and `bincount` with `weights` solves it directly.
`np.add.at(F, I, X)` also works if F already exists.

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

```python
w = h = 256
I = rng.integers(0, 4, (h, w, 3), dtype=np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
print(len(colors))
```

`np.unique(..., axis=0)` handles this cleanly. Older solutions did a bit-packing
trick (`R*65536 + G*256 + B`) which is faster but harder to read.

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

```python
A = rng.integers(0, 10, (3, 4, 3, 4))
print(A.sum(axis=(-2, -1)))
```

`axis` takes a tuple. Negative indices mean you don't need to know the rank.

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)

```python
D = rng.uniform(0, 1, 100)
S = rng.integers(0, 10, 100)
means = np.bincount(S, weights=D) / np.bincount(S)
```

Sum per group divided by count per group. This is a groupby-mean in two calls.

#### 69. How to get the diagonal of a dot product? (★★★)

```python
A = rng.uniform(0, 1, (5, 5))
B = rng.uniform(0, 1, (5, 5))

np.einsum("ij,ji->i", A, B)
```

The naive `np.diag(A @ B)` computes all 25 entries and throws away 20 of them.
The einsum version only computes the 5 you want.

#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

```python
Z = np.array([1, 2, 3, 4, 5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z) - 1) * nz)
Z0[::nz + 1] = Z
```

Allocate the full-size zero array, then drop the values in at every 4th slot.

#### 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)? (★★★)

```python
A = np.ones((5, 5, 3))
B = 2 * np.ones((5, 5))
print(A * B[:, :, None])
```

Broadcasting aligns from the right, so (5,5) vs (5,5,3) doesn't line up. Adding
a trailing axis with `None` makes B (5,5,1), which then stretches across the 3.

#### 72. How to swap two rows of an array? (★★★)

```python
A = np.arange(25).reshape(5, 5)
A[[0, 1]] = A[[1, 0]]
```

Fancy indexing on the right builds a copy first, so this doesn't clobber itself
the way a naive `A[0] = A[1]; A[1] = A[0]` would.

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)

```python
faces = rng.integers(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F) * 3, 2)
F = np.sort(F, axis=1)
G = np.unique(F, axis=0)
```

Each triangle (a,b,c) becomes edges (a,b), (b,c), (c,a). Sorting each pair means
(a,b) and (b,a) count as the same edge before deduping.

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

```python
C = np.array([0, 1, 3, 2])
A = np.repeat(np.arange(len(C)), C)
```

The inverse of `bincount`: repeat each index as many times as its count.

#### 75. How to compute averages using a sliding window over an array? (★★★)

```python
from numpy.lib.stride_tricks import sliding_window_view

Z = np.arange(20)
print(sliding_window_view(Z, 3).mean(axis=-1))
```

`sliding_window_view` gives a *view*, so no data is copied. Before it existed you
had to do this with a cumsum trick:

```python
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
```

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)

```python
from numpy.lib.stride_tricks import sliding_window_view

Z = np.arange(10)
print(sliding_window_view(Z, 3))
```

Same tool as #75. The old way was to hand-build strides with
`as_strided`, which works but is easy to get catastrophically wrong (you can
read past the end of the buffer with no error).

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

```python
Z = rng.integers(0, 2, 100, dtype=bool)
np.logical_not(Z, out=Z)

Z = rng.uniform(-1.0, 1.0, 100)
np.negative(Z, out=Z)
```

Both are ufuncs, so `out=` gives you the in-place version.

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T ** 2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] +
          (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - p
    return np.sqrt((D ** 2).sum(axis=1))

P0 = rng.uniform(-10, 10, (10, 2))
P1 = rng.uniform(-10, 10, (10, 2))
p  = rng.uniform(-10, 10, (1, 2))
print(distance(P0, P1, p))
```

This is distance to the infinite line, not the segment. Project p onto the line
(that's `U`), find that closest point, measure. If you wanted segment distance
you'd clip U to [0,1] first.

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

```python
P = rng.uniform(-10, 10, (10, 2))
print(np.array([distance(P0, P1, p_i) for p_i in P]))
```

Reuses #78 once per point. Not the fastest possible, but it's readable, and the
function already broadcasts over all the lines.

#### 80. Consider an arbitrary array, write a function that extracts a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

```python
Z = rng.integers(0, 10, (10, 10))
shape, fill, position = (5, 5), 0, (1, 1)

R = np.ones(shape, dtype=Z.dtype) * fill
P = np.array(position)
Rs = np.array(R.shape)
Zs = np.array(Z.shape)

R_start = np.zeros((len(shape),)).astype(int)
R_stop = np.array(shape).astype(int)
Z_start = P - Rs // 2
Z_stop = P + Rs // 2 + Rs % 2

R_start = (R_start - np.minimum(Z_start, 0)).tolist()
Z_start = np.maximum(Z_start, 0).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop - Zs, 0))).tolist()
Z_stop = np.minimum(Z_stop, Zs).tolist()

r = tuple(slice(start, stop) for start, stop in zip(R_start, R_stop))
z = tuple(slice(start, stop) for start, stop in zip(Z_start, Z_stop))
R[r] = Z[z]
print(R)
```

Genuinely fiddly. The idea: build the output prefilled, then work out which slice
of the source maps onto which slice of the output once you clip at the edges.
In real code I'd just `np.pad` the source first and slice normally — much easier
to convince yourself it's correct.

#### 81. Consider an array Z = [1,2,...,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], ..., [11,12,13,14]]? (★★★)

```python
Z = np.arange(1, 15)
R = sliding_window_view(Z, 4)
```

#### 82. Compute a matrix rank (★★★)

```python
Z = rng.uniform(0, 1, (10, 10))
print(np.linalg.matrix_rank(Z))
```

Under the hood it's an SVD, counting singular values above a tolerance — rank
is a numerical question, not an exact one, once floats are involved.

#### 83. How to find the most frequent value in an array?

```python
Z = rng.integers(0, 10, 50)
print(np.bincount(Z).argmax())
```

Only works for non-negative ints. For anything else, `np.unique(Z, return_counts=True)`
and take the argmax of the counts.

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

```python
Z = rng.integers(0, 5, (10, 10))
C = sliding_window_view(Z, (3, 3))
print(C.shape)   # (8, 8, 3, 3)
```

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

```python
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super().__setitem__((i, j), value)
        super().__setitem__((j, i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(rng.integers(0, 10, (5, 5)))
S[2, 3] = 42
print(S)
```

Every write mirrors itself. The helper also symmetrizes the initial data —
`Z + Z.T` would double the diagonal, hence subtracting it back off.

#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (★★★)

```python
p, n = 10, 20
M = np.ones((p, n, n))
V = np.ones((p, n, 1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S.shape)   # (n, 1)
```

`tensordot` contracts over the p axis and the n axis simultaneously. The einsum
version reads more clearly to me: `np.einsum("pij,pjk->ik", M, V)`.

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

```python
Z = np.ones((16, 16))
k = 4
S = Z.reshape(16 // k, k, 16 // k, k).sum(axis=(1, 3))
print(S.shape)   # (4, 4)
```

The reshape splits each axis into (block index, position within block), then you
sum away the two within-block axes. No copying involved.

#### 88. How to implement the Game of Life using numpy arrays? (★★★)

```python
def iterate(Z):
    # count the 8 neighbours of every cell by summing 8 shifted views
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2]                 + Z[1:-1, 2:] +
         Z[2:  , 0:-2] + Z[2:  , 1:-1] + Z[2:  , 2:])

    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)

    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z

Z = rng.integers(0, 2, (50, 50))
for _ in range(100):
    Z = iterate(Z)
```

The whole trick is counting neighbours with eight offset slices instead of
looping over cells. The border is left dead so the slices stay in bounds.

#### 89. How to get the n largest values of an array (★★★)

```python
Z = np.arange(10000)
rng.shuffle(Z)
n = 5

print(Z[np.argpartition(-Z, n)[:n]])
```

`argpartition` is O(n) and only guarantees the top-n are in the first n slots —
unordered. If you need them sorted, sort just that small slice afterwards. Beats
a full `argsort` on a big array.

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combination of every item) (★★★)

```python
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)

print(cartesian(([1, 2, 3], [4, 5], [6, 7])))
```

`meshgrid` with `indexing="ij"` then flatten each grid. `itertools.product` is
clearer if you don't need it as an array.

#### 91. How to create a record array from a regular array? (★★★)

```python
Z = np.array([("Hello", 2.5, 3), ("World", 3.6, 2)])
R = np.rec.fromarrays(Z.T, names="col1, col2, col3",
                      formats="S8, f8, i8")
print(R.col1)
```

Note `np.rec` — `np.core.records` from the old solutions is deprecated in numpy 2.

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

```python
Z = rng.random(int(5e7))

Z ** 3
Z * Z * Z
np.power(Z, 3)
np.einsum("i,i,i->i", Z, Z, Z)
```

Worth actually timing these. `Z*Z*Z` and einsum tend to beat `Z**3` / `np.power`,
because the general power function has to handle fractional exponents and can't
just do two multiplications.

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

```python
A = rng.integers(0, 5, (8, 3))
B = rng.integers(0, 5, (2, 2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3, 1)).all(1))[0]
print(rows)
```

Broadcast every element of A against every element of B, then ask: for each row
of B, does *some* element of the A row match (`any`), and does that hold for
*every* row of B (`all`).

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)

```python
Z = rng.integers(0, 5, (10, 3))
E = np.all(Z[:, 1:] == Z[:, :-1], axis=1)
U = Z[~E]
print(U)
```

`E` marks rows where every neighbouring pair is equal (i.e. all three the same);
negate it to keep the rest.

#### 95. Convert a vector of ints into a matrix binary representation (★★★)

```python
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
print(B[:, ::-1])
```

Bitwise-and each value against every power of two, then reverse so the most
significant bit reads first. `np.unpackbits` does this for uint8 directly.

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)

```python
Z = rng.integers(0, 2, (6, 3))
print(np.unique(Z, axis=0))
```

The `axis=0` argument makes this a one-liner now. Older solutions had to view
the rows as a compound dtype to make them comparable as single items.

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

```python
A = rng.uniform(0, 1, 10)
B = rng.uniform(0, 1, 10)

np.einsum("i->", A)          # np.sum(A)
np.einsum("i,i->i", A, B)    # A * B
np.einsum("i,i", A, B)       # np.inner(A, B)
np.einsum("i,j->ij", A, B)   # np.outer(A, B)
```

Reading the notation: repeated index = multiply together, index missing from the
right of `->` = sum over it. Everything else follows from those two rules.

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

```python
phi = np.arange(0, 10 * np.pi, 0.1)
a = 1
x = a * phi * np.cos(phi)
y = a * phi * np.sin(phi)

dr = np.hypot(np.diff(x), np.diff(y))   # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                   # cumulative arc length

r_int = np.linspace(0, r.max(), 200)    # evenly spaced along the path
x_int = np.interp(r_int, r, x)
y_int = np.interp(r_int, r, y)
```

The insight is that "equidistant along the curve" means evenly spaced in
*cumulative arc length*, not in the parameter. So compute arc length, then
interpolate x and y against it.

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

```python
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4

M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```

Two conditions: `mod(X, 1) == 0` catches "is a whole number" even though the
dtype is float, and the row must sum to n.

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (★★★)

```python
X = rng.standard_normal(100)
N = 1000

idx = rng.integers(0, X.size, (N, X.size))   # resample with replacement
means = X[idx].mean(axis=1)                  # mean of each resample
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```

Nice one to end on, because it's a real statistical technique rather than an
indexing trick: resample the data with replacement N times, take the mean of
each, and read the 2.5th and 97.5th percentiles off the resulting distribution.
No assumption of normality needed. Building the whole (N, size) index matrix at
once means the entire bootstrap is vectorized — no Python loop.
