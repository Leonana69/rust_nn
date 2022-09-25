use core::panic;
use std::{ops::{Mul, AddAssign, Add, Index, IndexMut}, process::exit};
use rand::{Rng, prelude::Distribution, distributions::Standard};
#[derive(Clone, Debug)]
pub struct Array<T> {
    pub shape: Box<[usize]>,
    pub sub_size: Box<[usize]>,
    pub data: Box<[T]>,
}

impl Array<f64> {
    pub fn random(shape_: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let (shape, sub_size) = Self::parse_shape(shape_);

        let data = (0..sub_size[0]).map(|_| rng.gen::<f64>() - 0.5).collect();
        Array { shape, sub_size, data }
    }
}

impl Array<f32> {
    pub fn random(shape_: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let (shape, sub_size) = Self::parse_shape(shape_);

        let data = (0..sub_size[0]).map(|_| rng.gen::<f32>() - 0.5).collect();
        Array { shape, sub_size, data }
    }
}

impl<T> Array<T>
where
    T: Copy + Clone + From<f32> + Mul<T, Output = T> + AddAssign<T> + Add<Output = T>,
    Standard: Distribution<T>,
{
    pub fn empty() -> Self {
        Array { shape: Box::default(), sub_size: Box::default(), data: Box::default() }
    }

    pub fn zeros(shape_: &[usize]) -> Self {
        Self::fill(shape_, T::from(0.0))
    }

    fn parse_shape(shape_: &[usize]) -> (Box<[usize]>, Box<[usize]>) {
        let mut shape: Vec<usize> = shape_.to_vec();
        let mut sub_size: Vec<usize> = Vec::new();
        let mut size = shape.iter().product();
        if shape.len() == 1 {
            shape.push(1);
        }

        for i in 0..shape.len() {
            sub_size.push(size);
            size /= shape[i];
        }

        (shape.into_boxed_slice(), sub_size.into_boxed_slice())
    }

    pub fn with(shape_: &[usize], data: &[T]) -> Self {
        let (shape, sub_size) = Self::parse_shape(shape_);
        if sub_size[0] != data.len() {
            panic!("Init with data [FAILED]");
        } else {
            Array { shape, sub_size, data: data.into() }
        }
    }

    pub fn fill(shape_: &[usize], value: T) -> Self {
        let (shape, sub_size) = Self::parse_shape(shape_);
        let size = sub_size[0];
        Array { shape, sub_size, data: vec![value; size].into_boxed_slice() }
    }

    pub fn dot(&self, b: &Array<T>) -> Array<T> {
        let len_a = self.shape.len();
        let len_b = b.shape.len();
        if self.shape[len_a - 1].ne(&b.shape[len_b - 2]) {
            panic!("Shapes not aligned {} (dim {}) != {} (dim {})",
                self.shape[len_a - 1], len_a - 1, b.shape[len_b - 2], len_b - 2);
        }

        let mut temp = Self::zeros(&[&self.shape[..len_a - 1], &b.shape[..len_b - 2], &[b.shape[len_b - 1]]].concat());

        let cnt_a = self.sub_size[0] / self.sub_size[self.sub_size.len() - 2];
        let cnt_b = b.sub_size[0] / b.sub_size[b.sub_size.len() - 2];
        let ss_a = temp.sub_size[self.shape.len() - 2];
        let si_a = temp.sub_size[self.shape.len() - 1];
        let ss_b = temp.sub_size[self.shape.len() + b.shape.len() - 3];
        let sss_a = self.sub_size[self.sub_size.len() - 2];
        let sss_b = b.sub_size[b.sub_size.len() - 2];

        let rows_a = self.shape[len_a - 2];
        let cols_a = self.shape[len_a - 1];
        let cols_b = b.shape[len_b - 1];

        // println!("{:?} x {:?}", self.shape, b.shape);

        for ca in 0..cnt_a {
            for i in 0..rows_a {
                for cb in 0..cnt_b {
                    for j in 0..cols_b {
                        let index = ca * ss_a + i * si_a + cb * ss_b + j;
                        let mut s = T::from(0.0);
                        for k in 0..cols_a {
                            s += self.data[ca * sss_a + i * cols_a + k] * b.data[cb * sss_b + k * cols_b + j];
                        }
                        temp.data[index] = s;
                    }
                }
            }
        }
        temp
    }

    pub fn to_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    pub fn t(&self) -> Array<T> {
        if self.shape.len() > 2 {
            panic!("Unable to transpose for nd array");
        } else {
            let mut temp: Array<T> = Array::zeros(&[self.shape[1], self.shape[0]]);
            let rows = temp.shape[0];
            let cols = temp.shape[1];
            
            for i in 0..rows {
                for j in 0..cols {
                    temp.data[j * rows + i] = self.data[i * cols + j];
                }
            }
            temp
        }
    }

    pub fn add_v(&mut self, rhs: T) -> &Self {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] + rhs;
        }
        self
    }

    pub fn add_m(&mut self, rhs: &Array<T>) -> &Self {
        if self.data.len() != rhs.data.len() {
            println!("add_m: dim not match");
        } else {
            for i in 0..self.data.len() {
                self.data[i] = self.data[i] + rhs.data[i];
            }
        }
        self
    }

    pub fn mul_v(&mut self, rhs: T) -> &Self {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] * rhs;
        }
        self
    }
}

impl<T> Index<&[usize]> for Array<T>
where
    T: Copy + Clone + From<f32> + Mul<T, Output = T> + AddAssign<T> + Add<Output = T>,
    Standard: Distribution<T>,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let mut id: usize = 0;
        for i in 0..self.shape.len() {
            if self.shape[i] > index[i] {
                if i == self.shape.len() - 1 {
                    id += index[i];
                } else {
                    id += index[i] * self.sub_size[i + 1];
                }
            } else {
                panic!("Indexing out of range");
            }
        }

        &self.data[id]
    }
}

impl<T> IndexMut<&[usize]> for Array<T>
where
    T: Copy + Clone + From<f32> + Mul<T, Output = T> + AddAssign<T> + Add<Output = T>,
    Standard: Distribution<T>,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let mut id: usize = 0;
        for i in 0..self.shape.len() {
            if self.shape[i] > index[i] {
                if i == self.shape.len() - 1 {
                    id += index[i];
                } else {
                    id += index[i] * self.sub_size[i + 1];
                }
            } else {
                panic!("Indexing out of range");
            }
        }

        &mut self.data[id]
    }
}