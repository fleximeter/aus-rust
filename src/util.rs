//! # Utilities
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
/// This function will panic if `vec2` is different in length than `vec1`.
#[inline]
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    assert_eq!(vec1.len(), vec2.len());
    let mut sum = 0.0;
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

/// Generates a vector of floats that are evenly spaced, beginning at `start_val` and ending on `end_val`.
#[inline]
pub fn linspace(start_val: f64, end_val: f64, size: usize) -> Vec<f64> {
    let mut scale: Vec<f64> = vec![0.0; size];
    let slope = (end_val - start_val) / (size + 1) as f64;
    for i in 0..size {
        scale[i] = start_val + slope * i as f64;
    }
    scale
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
                val += vec[i].abs() * vec[i].abs();
            }
            f64::sqrt(val)
        }
    }
}

/// A simple max function that also returns the argmax
#[inline(always)]
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

/// A simple max function that also returns the argmax
#[inline(always)]
pub fn max<T: std::cmp::PartialOrd + Copy>(vec: &[T]) -> Option<T> {
    if vec.len() == 0 {
        return None;
    } else {
        let mut max_val: T = vec[0];
        for i in 1..vec.len() {
            if max_val < vec[i] {
                max_val = vec[i];
            }
        }
        return Some(max_val);
    }
}

/// A simple min function that also returns the argmin
#[inline(always)]
pub fn minargmin<T: std::cmp::PartialOrd + Copy>(vec: &[T]) -> Option<(T, usize)> {
    if vec.len() == 0 {
        return None;
    } else {
        let mut min_idx: usize = 0;
        let mut min_val: T = vec[0];
        for i in 1..vec.len() {
            if min_val > vec[i] {
                min_val = vec[i];
                min_idx = i;
            }
        }
        return Some((min_val, min_idx));
    }
}

/// A simple min function
#[inline(always)]
pub fn min<T: std::cmp::PartialOrd + Copy>(vec: &[T]) -> Option<T> {
    if vec.len() == 0 {
        return None;
    } else {
        let mut min_val: T = vec[0];
        for i in 1..vec.len() {
            if min_val > vec[i] {
                min_val = vec[i];
            }
        }
        return Some(min_val);
    }
}

/// A function that searches an ordered slice in log(n) time
pub fn ordered_search<T: std::cmp::PartialOrd + std::ops::Sub<Output=T> + Copy>(vec: &[T], target: T) -> Option<usize> {
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
                return Some(middle_idx);
            } else if upper_idx - lower_idx == 1 {
                if vec[upper_idx] - target >= target - vec[lower_idx] {
                    return Some(lower_idx);
                } else {
                    return Some(upper_idx);
                }
            } else if vec[middle_idx] > target {
                upper_idx = middle_idx;
            } else {
                lower_idx = middle_idx;
            }
            middle_idx = lower_idx + (upper_idx - lower_idx) / 2;
        }
    }
}

/// A function that searches an ordered slice in log(n) time.
/// This function finds the index of the value that is closest to `target`,
/// but still <= `target`.
pub fn ordered_search_le<T: std::cmp::PartialOrd + std::ops::Sub<Output=T> + Copy>(vec: &[T], target: T) -> Option<usize> {
    if vec.len() == 0 {
        return None;
    } else if vec[0] > target {
        return None;
    } else if vec[0] == target {
        return Some(0);
    } else if vec[vec.len() - 1] <= target {
        return Some(vec.len() - 1);
    } else {
        let mut lower_idx: usize = 0;
        let mut upper_idx: usize = vec.len() - 1;
        let mut middle_idx: usize = upper_idx / 2;
        loop {
            if vec[middle_idx] == target {
                return Some(middle_idx);
            } else if upper_idx - lower_idx < 2 {
                return Some(lower_idx);
            } else if target < vec[middle_idx] {
                upper_idx = middle_idx;
            } else {
                lower_idx = middle_idx;
            }
            middle_idx = lower_idx + (upper_idx - lower_idx) / 2;
        }
    }
}

#[inline(always)]
pub fn wrap(val: f64, lower_bound: f64, upper_bound: f64) -> f64 {
    let adjusted_upper = upper_bound - lower_bound;
    let adjusted_val = val - lower_bound;
    let mut wrapped = adjusted_val % adjusted_upper;
    if wrapped < 0.0 {
        wrapped += adjusted_upper;
    }
    wrapped + lower_bound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max() {
        // test empty vector
        let vec0: Vec<i32> = vec![];
        assert_eq!(max(&vec0), None);

        // test integer vectors
        let vec1: Vec<i32> = vec![-9, -1, -593, -54, -492, -491, -85];
        assert_eq!(max(&vec1), Some(-1));
        let vec2: Vec<i32> = vec![4, 9, 5, 492, 491, 85];
        assert_eq!(max(&vec2), Some(492));
        let vec3: Vec<i32> = vec![4, 9, 5, 492, 491, 85, 1424];
        assert_eq!(max(&vec3), Some(1424));
        let vec4: Vec<i32> = vec![5953, 9, 5, 492, 491, 85];
        assert_eq!(max(&vec4), Some(5953));
    }

    #[test]
    fn test_maxargmax() {
        // test empty vector
        let vec0: Vec<i32> = vec![];
        assert_eq!(maxargmax(&vec0), None);

        // test integer vectors
        let vec1: Vec<i32> = vec![-9, -1, -593, -54, -492, -491, -85];
        assert_eq!(maxargmax(&vec1), Some((-1, 1)));
        let vec2: Vec<i32> = vec![4, 9, 5, 492, 491, 85];
        assert_eq!(maxargmax(&vec2), Some((492, 3)));
        let vec3: Vec<i32> = vec![4, 9, 5, 492, 491, 85, 1424];
        assert_eq!(maxargmax(&vec3), Some((1424, 6)));
        let vec4: Vec<i32> = vec![5953, 9, 5, 492, 491, 85];
        assert_eq!(maxargmax(&vec4), Some((5953, 0)));
    }

    #[test]
    fn test_min() {
        // test empty vector
        let vec0: Vec<i32> = vec![];
        assert_eq!(min(&vec0), None);

        // test integer vectors
        let vec1: Vec<i32> = vec![-4, 9, 0, 5, -492, -491, 85];
        assert_eq!(min(&vec1), Some(-492));
        let vec2: Vec<i32> = vec![4, 9, 5, 492, 491, 85];
        assert_eq!(min(&vec2), Some(4));
        let vec3: Vec<i32> = vec![4, 9, 5, 492, 491, 85, 1];
        assert_eq!(min(&vec3), Some(1));
        let vec4: Vec<i32> = vec![59, 9, 5, 492, 491, 85];
        assert_eq!(min(&vec4), Some(5));
    }

    #[test]
    fn test_minargmin() {
        // test empty vector
        let vec0: Vec<i32> = vec![];
        assert_eq!(minargmin(&vec0), None);

        // test integer vectors
        let vec1: Vec<i32> = vec![-4, 9, 0, 5, -492, -491, 85];
        assert_eq!(minargmin(&vec1), Some((-492, 4)));
        let vec2: Vec<i32> = vec![4, 9, 5, 492, 491, 85];
        assert_eq!(minargmin(&vec2), Some((4, 0)));
        let vec3: Vec<i32> = vec![4, 9, 5, 492, 491, 85, 1];
        assert_eq!(minargmin(&vec3), Some((1, 6)));
        let vec4: Vec<i32> = vec![59, 9, 5, 492, 491, 85];
        assert_eq!(minargmin(&vec4), Some((5, 2)));
    }

    #[test]
    fn test_ordered_search() {
        // test empty vector
        let vec0: Vec<i32> = vec![];

        // test integer vector
        let vec1: Vec<i32> = vec![2, 5, 8, 9, 11, 13, 15, 19, 21, 30, 50, 87, 90];
        assert_eq!(ordered_search(&vec0, 0), None);
        assert_eq!(ordered_search(&vec1, 0), Some(0));
        assert_eq!(ordered_search(&vec1, 2), Some(0));
        assert_eq!(ordered_search(&vec1, 15), Some(6));
        assert_eq!(ordered_search(&vec1, 13), Some(5));
        assert_eq!(ordered_search(&vec1, 14), Some(5));
        assert_eq!(ordered_search(&vec1, 31), Some(9));
        assert_eq!(ordered_search(&vec1, 7), Some(2));
        assert_eq!(ordered_search(&vec1, 49), Some(10));
        assert_eq!(ordered_search(&vec1, 4), Some(1));
        assert_eq!(ordered_search(&vec1, 90), Some(12));
        assert_eq!(ordered_search(&vec1, 89), Some(12));

        // test float vector
        let vec2: Vec<f64> = vec![2.0, 5.0, 8.1, 9.7, 11.1, 13.0, 15.0, 19.5, 21.99, 30.423, 50.2435, 87.698, 90.382];
        assert_eq!(ordered_search(&vec2, 0.0), Some(0));
        assert_eq!(ordered_search(&vec2, 2.0), Some(0));
        assert_eq!(ordered_search(&vec2, 15.0), Some(6));
        assert_eq!(ordered_search(&vec2, 11.1), Some(4));
        assert_eq!(ordered_search(&vec2, 30.425), Some(9));
        assert_eq!(ordered_search(&vec2, 8.1), Some(2));
        assert_eq!(ordered_search(&vec2, 10329.023), Some(12));
        assert_eq!(ordered_search(&vec2, 21.99), Some(8));
        assert_eq!(ordered_search(&vec2, 41.3294), Some(10));
        assert_eq!(ordered_search(&vec2, 2.01), Some(0));
        assert_eq!(ordered_search(&vec2, 12.8889), Some(5));
        assert_eq!(ordered_search(&vec2, 23.0), Some(8));
    }

    #[test]
    fn test_ordered_search_le() {
        let vec0: Vec<i32> = vec![];
        let vec1: Vec<i32> = vec![2, 5, 8, 9, 11, 13, 15, 19, 21, 30, 50, 87, 90];
        assert_eq!(ordered_search_le(&vec0, 0), None);
        assert_eq!(ordered_search_le(&vec1, 0), None);
        assert_eq!(ordered_search_le(&vec1, 2), Some(0));
        assert_eq!(ordered_search_le(&vec1, 15), Some(6));
        assert_eq!(ordered_search_le(&vec1, 13), Some(5));
        assert_eq!(ordered_search_le(&vec1, 14), Some(5));
        assert_eq!(ordered_search_le(&vec1, 31), Some(9));
        assert_eq!(ordered_search_le(&vec1, 7), Some(1));
        assert_eq!(ordered_search_le(&vec1, 49), Some(9));
        assert_eq!(ordered_search_le(&vec1, 4), Some(0));
        assert_eq!(ordered_search_le(&vec1, 90), Some(12));
        assert_eq!(ordered_search_le(&vec1, 89), Some(11));
    }

    #[test]
    fn test_wrap() {
        const EPSILON: f64 = 1e-8;
        assert!({
            let val = 1.1000001;
            let target = val;
            let modval = wrap(val, 1.1, 4.1);
            target - EPSILON < modval && target + EPSILON > modval
        });
        assert!({
            let val = 4.099999;
            let target = val;
            let modval = wrap(val, 1.1, 4.1);
            target - EPSILON < modval && target + EPSILON > modval
        });
        assert!({
            let val = 2.4231;
            let target = val;
            let modval = wrap(val, 1.1, 4.1);
            target - EPSILON < modval && target + EPSILON > modval
        });
        assert!({
            let val = 0.0;
            let target = 3.0;
            let modval = wrap(val, 1.1, 4.1);
            target - EPSILON < modval && target + EPSILON > modval
        });
        assert!({
            let val = -15.3;
            let target = 2.7;
            let modval = wrap(val, 1.1, 4.1);
            target - EPSILON < modval && target + EPSILON > modval
        });
        assert!({
            let val = 15.3;
            let target = 3.3;
            let modval = wrap(val, 1.1, 4.1);
            target - EPSILON < modval && target + EPSILON > modval
        });
    }
}
