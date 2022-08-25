pub struct Array {
    pub dim_in: usize,
    pub dim_out: usize,
    pub data: Box<[f64]>,
}

impl Array {
    pub fn empty(dim_in: usize, dim_out: usize) -> Self {
        Array {
            dim_in,
            dim_out,
            data: vec![0 as f64; dim_in * dim_out].into_boxed_slice()
        }
    }
}