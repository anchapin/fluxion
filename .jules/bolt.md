## 2024-05-24 - [Fluxion Performance Optimization]
**Learning:** `VectorField` operations in `fluxion` are not fully in-place. The arithmetic operators (`Add`, `Sub`, `Mul`, `Div`) take `self` by value and reuse the buffer, which is good. However, operations like `map` and `zip_with` allocate new `Vec`s.

The `map` function allocates a new `VectorField` and iterates using `map(f).collect()`. This is standard but can be optimized if we allow in-place mutation.

**Action:** Implement `map_in_place` or `apply` for `VectorField` to allow reusing the buffer when applying a function element-wise. Also, check if `zip_with` can be optimized or if we can use a more iterator-based approach to avoid intermediate allocations in complex expressions.

However, `ContinuousTensor` trait requires `map` to return `Self`. Changing the trait might be a breaking change or require significant refactoring.

Wait, `VectorField`'s `Add` implementation:
```rust
impl Add for VectorField {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        // Optimization: reuse self buffer
        assert_eq!(self.len(), rhs.len(), "Tensor dimension mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
        self
    }
}
```
This reuses `self`'s buffer. But what about `&self + &other`? The trait bounds in `ContinuousTensor` say `Add<Output=Self>`. `fluxion` seems to use `clone()` a lot before operations (e.g. `v1.clone() + v2.clone()`).

The benchmarks show that `vector_map` is slightly slower than `raw_map` (3.9us vs 3.1us). `raw_map` allocates too (the benchmark collects into a Vec). So `vector_map` overhead is likely just the wrapper struct construction?

Let's look at `src/sim/engine.rs` again.
In `step_physics_5r1c`:
```rust
        // Optimization: Avoid creating t_e vector. Use map with scalar outdoor_temp.
        // t_e - mass_temperatures = outdoor_temp - mass_temperatures
        let q_m_net = self.h_tr_em.clone() * self.mass_temperatures.map(|m| outdoor_temp - m)
            + self.h_tr_ms.clone() * (t_s_free - self.mass_temperatures.clone())
            + phi_m; // Add gain directly to mass node
```
`self.mass_temperatures.map(...)` creates a NEW vector.
`self.h_tr_em.clone()` creates a NEW vector.
The multiplication consumes the cloned `h_tr_em` (reusing its buffer) and the result of map.

There are many `clone()` calls.

One potential optimization: `VectorField::map` currently:
```rust
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        VectorField {
            data: self.data.iter().copied().map(f).collect(),
        }
    }
```
It always allocates.

If we have `impl ContinuousTensor`, we might add `map_into(self, f) -> Self`.
But `ContinuousTensor` trait defines `map(&self, f) -> Self`.

Wait, `step_physics` has this pattern:
```rust
        let q_m_net = self.h_tr_em.clone() * self.mass_temperatures.map(|m| outdoor_temp - m)
```

`self.mass_temperatures` is a field of `ThermalModel`. We cannot consume it here, so `map` (which takes `&self`) is correct in principle (we need a new vector).

However, `map` followed by `*` involves:
1. `map`: allocates result V1.
2. `h_tr_em.clone()`: allocates V2.
3. `V2 * V1`: reuses V2 buffer, drops V1 buffer.

Ideally, we want:
`result = h_tr_em * (outdoor_temp - mass_temperatures)`

We can do `zip_with`?
`self.h_tr_em.zip_with(&self.mass_temperatures, |h, m| h * (outdoor_temp - m))`
This would allocate ONCE (the result vector).

Currently:
1. `mass_temperatures.map(...)` -> allocates V_temp.
2. `h_tr_em.clone()` -> allocates V_h.
3. `V_h * V_temp` -> reuses V_h, frees V_temp.

So we have 2 allocations and 1 deallocation. `zip_with` would be 1 allocation.

Let's check `zip_with` implementation:
```rust
    fn zip_with<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        assert_eq!(self.len(), other.len(), "Tensor dimension mismatch");
        VectorField {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }
```
It allocates exactly once.

So replacing `clone() * map()` sequences with `zip_with` is a valid optimization.

Example in `step_physics_5r1c`:
```rust
let q_m_net = self.h_tr_em.clone() * self.mass_temperatures.map(|m| outdoor_temp - m)
```
Can be:
```rust
let q_m_net = self.h_tr_em.zip_with(&self.mass_temperatures, |h, m| h * (outdoor_temp - m))
```

Also:
```rust
self.h_tr_ms.clone() * (t_s_free - self.mass_temperatures.clone())
```
`t_s_free` is a local variable (owned `VectorField`). `self.mass_temperatures.clone()` allocates. `t_s_free - ...` reuses `t_s_free`.

If `t_s_free` is owned, we can do:
`t_s_free - &self.mass_temperatures`? No, `Sub` takes `Self`.
So `self.mass_temperatures.clone()` is needed if we use `Sub`.

However, we can use `map` or `zip_with` on `t_s_free`?
`t_s_free` is `ContinuousTensor`.

If we define `sub_assign` or similar?
The trait `ContinuousTensor` extends `Add<Output=Self>`, etc.

Wait, `VectorField` implements `Add` consuming self.
Does it implement `Add<&VectorField>`?
I don't see `impl Add<&VectorField> for VectorField` in `src/physics/cta.rs`.
Only `impl Add for VectorField`.

So `t_s_free - self.mass_temperatures.clone()` is necessary for the syntax `a - b`.

But we can do:
`t_s_free.zip_with(&self.mass_temperatures, |a, b| a - b)` -> This allocates a NEW vector!
The original `t_s_free` is dropped.
So `t_s_free - clone()`:
1. Clone `mass` -> alloc V_new.
2. `t_s_free - V_new` -> reuses `t_s_free`, drops V_new.
Total: 1 alloc, 1 free.

Using `zip_with`:
`t_s_free.zip_with(...)` -> alloc Result. `t_s_free` dropped.
Total: 1 alloc, 1 free.
Same cost.

BUT, `h_tr_ms.clone() * (...)`
`h_tr_ms` is a field. Clone allocates.
Then `*` reuses it.

Let's look at `q_m_net` calculation again.
```rust
        let q_m_net = self.h_tr_em.clone() * self.mass_temperatures.map(|m| outdoor_temp - m)
            + self.h_tr_ms.clone() * (t_s_free - self.mass_temperatures.clone())
            + phi_m; // Add gain directly to mass node
```

Term 1: `self.h_tr_em.clone() * self.mass_temperatures.map(...)`
- map: alloc V1.
- clone: alloc V2.
- mul: reuse V2, drop V1.
- result: V2.
Cost: 2 allocs.

Optimization 1: `self.h_tr_em.zip_with(&self.mass_temperatures, |h, m| h * (outdoor_temp - m))`
- zip_with: alloc V_res.
- result: V_res.
Cost: 1 alloc.

Term 2: `self.h_tr_ms.clone() * (t_s_free - self.mass_temperatures.clone())`
- `mass.clone()`: alloc V3.
- `t_s_free` (owned) - V3: reuse `t_s_free`, drop V3. Result is `t_s_free` (mutated).
- `h_tr_ms.clone()`: alloc V4.
- V4 * `t_s_free`: reuse V4, drop `t_s_free`.
- result: V4.
Cost: 2 allocs (V3, V4).

Optimization 2:
Can we do `h_tr_ms * (t_s_free - mass)` in one pass?
`h_tr_ms.zip_with(&t_s_free, |h, t| h * (t - mass))`?
We need access to `mass` inside the closure.
`self.h_tr_ms.zip_with(&t_s_free, |h, t| h * (t - self.mass_temperatures[...]))`?
Zip with allows only 2 iterators. We need 3 (h, t, m).

If we implemented a `zip_with3`? Or general iterator support?

`VectorField` exposes `iter()`.
We can construct the result directly:
```rust
let q_m_net_data: Vec<f64> = self.h_tr_em.iter()
    .zip(self.mass_temperatures.iter())
    .zip(self.h_tr_ms.iter())
    .zip(t_s_free.iter())
    .zip(phi_m.iter())
    .map(|((((&h_em, &tm), &h_ms), &ts), &phi)| {
        h_em * (outdoor_temp - tm) + h_ms * (ts - tm) + phi
    })
    .collect();
let q_m_net = VectorField::new(q_m_net_data);
```
This allocates EXACTLY ONE vector for `q_m_net`.

Current implementation:
1. Term 1 (2 allocs)
2. Term 2 (2 allocs)
3. Term 1 + Term 2 (reuse Term 1 buffer, drop Term 2 buffer)
4. + phi_m (reuse buffer)
Total allocs: 4. (Plus intermediate drops).

Optimized implementation: 1 alloc.
This is a huge win for the hot loop.

This pattern appears in `step_physics_5r1c` and `step_physics_6r2c`.

`t_s_free` calculation also:
```rust
        let ts_num_free = self.h_tr_ms.clone() * self.mass_temperatures.clone()
            + self.h_tr_is.clone() * t_i_free.clone()
            + phi_st.clone();
```
1. `h_tr_ms.clone()` (alloc) * `mass.clone()` (alloc) -> reuse 1, drop 1. (2 allocs)
2. `h_tr_is.clone()` (alloc) * `t_i_free.clone()` (alloc) -> reuse 1, drop 1. (2 allocs)
3. Sum 1 + 2 -> reuse 1, drop 2.
4. + `phi_st.clone()` (alloc) -> reuse 1, drop 1. (1 alloc)
Total: 5 allocs.

Optimized:
```rust
let ts_num_free_data: Vec<f64> = self.h_tr_ms.iter()
    .zip(self.mass_temperatures.iter())
    .zip(self.h_tr_is.iter())
    .zip(t_i_free.iter())
    .zip(phi_st.iter())
    .map(|((((&h_ms, &tm), &h_is), &ti), &phi)| {
        h_ms * tm + h_is * ti + phi
    })
    .collect();
let ts_num_free = VectorField::new(ts_num_free_data);
```
Total: 1 alloc.

**Plan:**
1. Identify the hottest loop `step_physics_5r1c` (and 6r2c).
2. Replace chain of vector ops with manual iterator zipping to reduce allocations.
3. This is specific to `VectorField` backend. The code in `ThermalModel` is generic `T: ContinuousTensor`.

Ah! `ThermalModel` is generic. I cannot just use `iter()` and `zip` if `T` is generic `ContinuousTensor`.
`ContinuousTensor` does not expose `iter()`.

However, `VectorField` implements `ContinuousTensor`.
The `ThermalModel` struct definition:
```rust
pub struct ThermalModel<T: ContinuousTensor<f64>> { ... }
```
And the implementation:
```rust
impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]>> ThermalModel<T> { ... }
```
It requires `AsRef<[f64]>`. `VectorField` implements `AsRef<[f64]>`.
So I can access the slice using `as_ref()`.

If I use `as_ref()`, I am assuming the underlying data is `[f64]`. This breaks "pure" tensor abstraction but `ThermalModel` already requires `AsRef<[f64]>` for `get_temperatures` etc.

So I can do:
```rust
let slice_a = self.tensor_a.as_ref();
let slice_b = self.tensor_b.as_ref();
// ...
let result_vec: Vec<f64> = slice_a.iter().zip(slice_b.iter())... .collect();
let result = T::from(VectorField::new(result_vec));
```
`T` implements `From<VectorField>`.

Wait, `ThermalModel` implementation block in `src/sim/engine.rs`:
```rust
impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]>> ThermalModel<T> {
```
Yes, it has `From<VectorField>` bound.

So I can optimize `q_m_net` and `ts_num_free` calculations by accessing slices directly, performing the map/zip/reduce logic in a single pass, creating a `VectorField`, and converting to `T`.

This will drastically reduce allocations in the hot loop.

**Verification:**
I should verify that `step_physics` logic is indeed using `q_m_net` and `ts_num_free` heavily. Yes, they are part of the state update.

Let's verify `step_physics_5r1c` implementation details again.
