use core::panic;
use std::{ops::{Index, IndexMut}};
use rand::Rng;
#[derive(Clone, Debug)]
pub struct Array<T> {
    pub shape: Box<[usize]>,
    pub sub_size: Box<[usize]>,
    pub data: Box<[T]>,
}

macro_rules! new_float_impl_for_array {
    ($type:ident) => {
        #[allow(dead_code)]
        impl Array<$type> {
            pub fn random_default(shape_: &[usize]) -> Self {
                Self::random(shape_, -1.0, 1.0)
            }

            pub fn random(shape_: &[usize], low: $type, high: $type) -> Self {
                let mut rng = rand::thread_rng();
                let (shape, sub_size) = Self::parse_shape(shape_);
        
                let data = (0..sub_size[0]).map(|_| rng.gen_range(low..high)).collect();
                Array { shape, sub_size, data }
            }

            pub fn get_zero() -> $type { 0.0 }
        }
    }
}

macro_rules! new_int_impl_for_array {
    ($type:ident) => {
        #[allow(dead_code)]
        impl Array<$type> {
            pub fn random_default(shape_: &[usize]) -> Self {
                Self::random(shape_, -128, 127)
            }

            pub fn random(shape_: &[usize], low: $type, high: $type) -> Self {
                let mut rng = rand::thread_rng();
                let (shape, sub_size) = Self::parse_shape(shape_);
        
                let data = (0..sub_size[0]).map(|_| rng.gen_range(low..high)).collect();
                Array { shape, sub_size, data }
            }

            pub fn get_zero() -> $type { 0 }
        }
    }
}

macro_rules! new_impl_for_array {
    ($type:ident) => {
        #[allow(dead_code)]
        impl Array<$type> {
            pub fn empty() -> Self {
                Array { shape: Box::default(), sub_size: Box::default(), data: Box::default() }
            }

            pub fn zeros(shape_: &[usize]) -> Self {
                Self::fill(shape_, Self::get_zero())
            }

            fn parse_shape(shape_: &[usize]) -> (Box<[usize]>, Box<[usize]>) {
                // shape_: &[a] --> (shape: Box<[a, 1]>, sub_size: Box<[a, 1]>)
                // shape_: &[a, b, c, ...] --> (shape: Box<[a, b, c, ...]>, sub_size: Box<[a*b*c*..., b*c*..., c*..., ...]>)
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

            pub fn with(shape_: &[usize], data: &[$type]) -> Self {
                let (shape, sub_size) = Self::parse_shape(shape_);
                if sub_size[0] != data.len() {
                    panic!("Init with data [FAILED]");
                } else {
                    Array { shape, sub_size, data: data.into() }
                }
            }
        
            pub fn fill(shape_: &[usize], value: $type) -> Self {
                let (shape, sub_size) = Self::parse_shape(shape_);
                let size = sub_size[0];
                Array { shape, sub_size, data: vec![value; size].into_boxed_slice() }
            }

            pub fn to_vec(self) -> Vec<$type> {
                self.data.into_vec()
            }

            /* matrix multiplication 
             * [a1, a2, ..., an] * [b1, b2, ..., bm] --> [a1, ..., a(n-1), b1, ..., b(m-2), bm]
             */
            pub fn dot(&self, b: &Array<$type>) -> Array<$type> {
                let len_a = self.shape.len();
                let len_b = b.shape.len();
                if self.shape[len_a - 1].ne(&b.shape[len_b - 2]) {
                    panic!("Shapes not aligned {} (dim {}) != {} (dim {})",
                        self.shape[len_a - 1], len_a - 1, b.shape[len_b - 2], len_b - 2);
                }
        
                let mut temp: Array<$type> = Self::zeros(&[&self.shape[..len_a - 1], &b.shape[..len_b - 2], &[b.shape[len_b - 1]]].concat());
        
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
        
                for ca in 0..cnt_a {
                    for i in 0..rows_a {
                        for cb in 0..cnt_b {
                            for j in 0..cols_b {
                                let index = ca * ss_a + i * si_a + cb * ss_b + j;
                                let mut s = Self::get_zero();
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

            /* transpose */
            pub fn t(&self) -> Array<$type> {
                if self.shape.len() > 2 {
                    panic!("Unable to transpose for nd array");
                } else {
                    let mut temp: Array<$type> = Self::zeros(&[self.shape[1], self.shape[0]]);
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
            
            /* add a constant value */
            pub fn add_v(&mut self, rhs: $type) -> &Self {
                for i in 0..self.data.len() {
                    self.data[i] = self.data[i] + rhs;
                }
                self
            }
            
            /* element wise add */
            pub fn add_m(&mut self, rhs: &Array<$type>) -> &Self {
                if self.data.len() != rhs.data.len() {
                    println!("add_m: dim not match");
                } else {
                    for i in 0..self.data.len() {
                        self.data[i] = self.data[i] + rhs.data[i];
                    }
                }
                self
            }
            
            /* multiplied by a constant */
            pub fn mul_v(&mut self, rhs: $type) -> &Self {
                for i in 0..self.data.len() {
                    self.data[i] = self.data[i] * rhs;
                }
                self
            }
        }

        impl Index<&[usize]> for Array<$type> {
            type Output = $type;
            fn index(&self, index: &[usize]) -> &Self::Output {
                let mut id: usize = 0;
                // check index
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

        impl IndexMut<&[usize]> for Array<$type> {
            fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
                let mut id: usize = 0;
                // check index
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
    }
}

new_impl_for_array!(f64);
new_impl_for_array!(i32);
new_float_impl_for_array!(f64);
new_int_impl_for_array!(i32);