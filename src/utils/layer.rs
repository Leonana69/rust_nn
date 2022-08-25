use super::{ops::{ Sigmoid, ReLU, Operator }, shape::Array};

pub trait Layer {
    fn init(&mut self) {}
    fn forward_prop(&mut self, input: &[f64]) -> Box<[f64]>;
    fn backward_prop(&mut self, error: &[f64], learn_rate: f64) -> Box<[f64]>;
}

#[derive(Default)]
pub struct Sequential  {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential  {
    pub fn new() -> Self {
        Sequential::default()
    }

    pub fn add<L>(&mut self, layer: L) -> &mut Self
    where
        L: Layer + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

#[derive(Default)]
pub struct SigmoidLayer {
    op: Sigmoid,
}

impl Layer for SigmoidLayer {
    fn forward_prop(&mut self, input: &[f64]) -> Box<[f64]> {
        Sigmoid::calculate(input, Sigmoid::activation)
    }

    fn backward_prop(&mut self, error: &[f64], learn_rate: f64) -> Box<[f64]> {
        let mut vec = Sigmoid::calculate(error, Sigmoid::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        vec
    }
}

pub struct DenseLayer {
    pub weights: Array,
    pub bias: Array,
}

impl DenseLayer {
    fn new(dim_in: usize, dim_out: usize) -> Self {
        DenseLayer {
            weights: Array::empty(dim_in, dim_out),
            bias: Array::empty(1, dim_out),
        }
    }

}

impl Layer for DenseLayer {
    fn forward_prop(&mut self, input: &[f64]) -> Box<[f64]> {
        let cols = self.weights.dim_out;
        let rows = self.weights.dim_in;
        let mut vec = vec![0 as f64; cols];
        for i in 0..rows {
            for j in 0..cols {
                vec[j] += self.weights.data[i * cols + j] * input[i];
            }
        }
        for j in 0..cols {
            vec[j] += self.bias.data[j];
        }
        vec.into_boxed_slice()
    }

    fn backward_prop(&mut self, error: &[f64], learn_rate: f64) -> Box<[f64]> {
        let cols = self.weights.dim_out;
        let rows = self.weights.dim_in;
        let mut vec = vec![0 as f64; cols];

        vec.into_boxed_slice()
    }
}
