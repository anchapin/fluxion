## 2024-05-23 - Reuse Buffer in Arithmetic Ops
**Learning:** When implementing arithmetic traits like `Add<Output=Self>` for structs wrapping `Vec`, taking `self` by value allows reusing the allocation if the implementation is changed from `zip().map().collect()` to `iter_mut().zip().for_each()`.
**Action:** Look for `zip().map().collect()` patterns in arithmetic implementations of owned types and replace with in-place mutation of `self`.
