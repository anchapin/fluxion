use napi::bindgen_prelude::Float64Array;

pub(crate) fn into_zero_copy_float64_array(mut data: Vec<f64>) -> Float64Array {
    let pointer = data.as_mut_ptr();
    let length = data.len();
    let capacity = data.capacity();
    std::mem::forget(data);

    unsafe {
        Float64Array::with_external_data(pointer, length, move |pointer, _| {
            drop(Vec::from_raw_parts(pointer, length, capacity));
        })
    }
}

#[napi_derive::napi]
pub fn transfer_matrix(matrix: Float64Array) -> Float64Array {
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_allocation_is_preserved_with_spare_capacity() {
        let mut data = Vec::with_capacity(16);
        data.extend([1.0, 2.0, 3.0, 4.0]);
        let pointer = data.as_ptr();

        let matrix = into_zero_copy_float64_array(data);

        assert_eq!(matrix.as_ptr(), pointer);
        assert_eq!(matrix.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matrix_round_trip_preserves_allocation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let pointer = data.as_ptr();
        let matrix = into_zero_copy_float64_array(data);

        let transferred = transfer_matrix(matrix);

        assert_eq!(transferred.as_ptr(), pointer);
        assert_eq!(transferred.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
