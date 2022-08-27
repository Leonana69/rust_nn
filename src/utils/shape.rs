use std::{ops::{Mul, AddAssign, Add}};
use rand::{Rng, prelude::Distribution, distributions::Standard};
use std::time::{Instant};
#[derive(Clone, Debug)]
pub struct Array<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl Array<f64> {
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..rows * cols).map(|_| rng.gen::<f64>() - 0.5).collect();
        Array { rows, cols, data }
    }
}

impl Array<i32> {
    // todo{}
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..rows * cols).map(|_| rng.gen::<i32>() - 0).collect();
        Array { rows, cols, data }
    }
}

impl<T> Array<T>
where
    T: Copy + Clone + From<i32> + Mul<T, Output = T> + AddAssign<T> + Add<Output = T>,
    Standard: Distribution<T>,
{
    pub fn empty(rows: usize, cols: usize) -> Self {
        Array { rows, cols, data: vec![T::from(0); rows * cols] }
    }

    // pub fn random(rows: usize, cols: usize) -> Self {
    //     let mut rng = rand::thread_rng();
    //     let data = (0..rows * cols).map(|_| rng.gen::<T>() - 0.5).collect();
    //     Array { rows, cols, data }
    // }

    pub fn fill(rows: usize, cols: usize, value: T) -> Self {
        Array { rows, cols, data: vec![value; rows * cols] }
    }

    pub fn dot(&self, b: &Array<T>) -> Array<T> {
        if self.cols != b.rows {
            Array::empty(0, 0)
        } else {
            let mut out: Array<T> = Array::empty(self.rows, b.cols);
            let cols_a = self.cols;
            let cols_b = b.cols;

            let out_slice = out.data.as_mut_slice();
            let self_slice = self.data.as_slice();
            let b_slice = b.data.as_slice();
    
            for i in 0..self.rows {
                for j in 0..cols_b {
                    let mut s = T::from(0);
                    for k in 0..cols_a {
                        s += self_slice[i * cols_a + k] * b_slice[k * cols_b + j];
                    }
                    out_slice[i * cols_b + j] = s;
                }
            }
            out
        }
    }

    pub fn to_vec(self) -> Vec<T> {
        self.data
    }

    pub fn t(&self) -> Array<T> {
        let mut temp: Array<T> = Array::empty(self.cols, self.rows);

        let temp_slice = temp.data.as_mut_slice();
        let self_slice = self.data.as_slice();

        for i in 0..self.rows {
            for j in 0..self.cols {
                temp_slice[j * self.rows + i] = self_slice[i * self.cols + j];
            }
        }
        temp
    }

    pub fn inc(&mut self, rhs: T) -> &Self {
        let self_slice = self.data.as_mut_slice();
        for i in 0..self.rows * self.cols {
            self_slice[i] = self_slice[i] + rhs;
        }
        self
    }

    pub fn add(&mut self, rhs: &Array<T>) -> &Self {
        let self_slice = self.data.as_mut_slice();
        let rhs_slice = rhs.data.as_slice();
        for i in 0..self.rows * self.cols {
            self_slice[i] = self_slice[i] + rhs_slice[i];
        }
        self
    }

    pub fn mul(&mut self, rhs: T) -> &Self {
        let self_slice = self.data.as_mut_slice();
        for i in 0..self.rows * self.cols {
            self_slice[i] = self_slice[i] * rhs;
        }
        self
    }
}

// impl<T: Add<Output = T>> Add for Array<T>
// where
//     T: Copy + Clone + From<i32> + Mul<T, Output = T> + AddAssign<T>,
//     Standard: Distribution<T>,
// {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         assert_eq!(self.rows, rhs.rows);
//         assert_eq!(self.cols, rhs.cols);
//         let mut temp: Array<T> = Array::empty(self.rows, self.cols);
//         for i in 0..self.rows {
//             for j in 0..self.cols {
//                 temp.data[i * self.cols + j] = self.data[i * self.cols + j] + rhs.data[i * self.cols + j];
//             }
//         }
//         temp
//     }
// }

// impl<T> Add<T> for Array<T>
// where
//     T: Copy + Clone + From<i32> + Mul<T, Output = T> + AddAssign<T> + Add<Output = T>,
//     Standard: Distribution<T>,
// {
//     type Output = Self;

//     fn add(self, rhs: T) -> Self::Output {
//         let mut temp: Array<T> = Array::empty(self.rows, self.cols);
//         for i in 0..self.rows {
//             for j in 0..self.cols {
//                 temp.data[i * self.cols + j] = self.data[i * self.cols + j] + rhs;
//             }
//         }
//         temp
//     }
// }

// impl<T> Mul<T> for Array<T>
// where
//     T: Copy + Clone + From<i32> + Mul<T, Output = T> + AddAssign<T> + Add<Output = T>,
//     Standard: Distribution<T>,
// {
//     type Output = Self;

//     fn mul(self, rhs: T) -> Self::Output {
//         let mut temp: Array<T> = Array::empty(self.rows, self.cols);
//         for i in 0..self.rows {
//             for j in 0..self.cols {
//                 temp.data[i * self.cols + j] = self.data[i * self.cols + j] * rhs;
//             }
//         }
//         temp
//     }
// }