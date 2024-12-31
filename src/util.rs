//! # Util
//! The `util` module contains utility functions for the other modules.

/// Represents a L1 or L2 norm.
pub enum Norm {
    L1,
    L2
}

/// Simple dot product function, implemented for code readability rather than using zip(), etc.
/// No vector length checks are performed - make sure that both vectors have the same length before
/// calling this function.
/// 
/// # Panics
/// This function will panic if `vec2` is shorter than `vec1`.
#[inline]
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

/// Computes the L1 or L2 norm of a vector
#[inline]
pub fn lnorm(vec: &[f64], norm_type: &Norm) -> f64 {
    match norm_type {
        Norm::L1 => {
            let mut val = 0.0;
            for i in 0..vec.len() {
                val += vec[i].abs();
            }
            val
        },
        Norm::L2 =>{
            let mut val = 0.0;
            for i in 0..vec.len() {
                val += vec[i] * vec[i];
            }
            f64::sqrt(val)
        }
    }
}

/// A simple max function that also returns the argmax
#[inline]
pub fn maxargmax<T: std::cmp::PartialOrd + Copy>(vec: &[T]) -> Option<(T, usize)> {
    if vec.len() == 0 {
        return None;
    } else {
        let mut max_idx: usize = 0;
        let mut max_val: T = vec[0];
        for i in 1..vec.len() {
            if max_val < vec[i] {
                max_val = vec[i];
                max_idx = i;
            }
        }
        return Some((max_val, max_idx));
    }
}

/// A function that searches an ordered slice in log(n) time
pub fn ordered_search<T: std::cmp::PartialOrd>(vec: &[T], target: T) -> Option<usize> {
    if vec.len() == 0 {
        return None;
    } else if vec[0] >= target {
        return Some(0);
    } else if vec[vec.len() - 1] <= target {
        return Some(vec.len() - 1);
    } else {
        let mut lower_idx: usize = 0;
        let mut upper_idx: usize = vec.len() - 1;
        let mut middle_idx: usize = upper_idx / 2;
        loop {
            if vec[middle_idx] == target {
                break;
            } else if lower_idx >= upper_idx {
                break;
            } else if vec[middle_idx] > target {
                upper_idx = middle_idx;
            } else {
                lower_idx = middle_idx;
            }
            middle_idx = lower_idx + (upper_idx - lower_idx) / 2;
        }
        return Some(lower_idx);
    }
}
