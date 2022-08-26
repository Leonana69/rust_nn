use super::{ops::{ Sigmoid, ReLU, Operator, calculate, TanH }, shape::Array, loss::MSE};
use crate::utils::loss::Loss;
pub trait Layer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64>;
    fn backward_prop(&mut self, error: &Vec<f64>, learning_rate: f64) -> Vec<f64>;
}

#[derive(Default)]
pub struct Sequential  {
    pub layers: Vec<Box<dyn Layer>>,
    pub learn_rate: f64,
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

    pub fn predict(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let len = input.len();
        let mut output: Vec<Vec<f64>> = Vec::default();
        for i in 0..len {
            let mut temp_input = &input[i];
            let mut temp_out: Vec<f64>;
            for l in self.layers.iter_mut() {
                temp_out = l.forward_prop(temp_input);
                temp_input = &temp_out;
            }
            output.push(temp_input.to_vec());
        }
        output
    }

    pub fn train(&mut self, input: &Vec<Vec<f64>>, truth: &Vec<Vec<f64>>, epochs: i32, learning_rate: f64) {
        let sample_len = input.len();
        for epoch in 0..epochs {
            let mut err = 0.0; // error on all samples
            for i in 0..sample_len {
                let mut temp_input = &input[i];
                let mut temp_out: Vec<f64>;
                for l in self.layers.iter_mut() {
                    temp_out = l.forward_prop(temp_input);
                    temp_input = &temp_out;
                }
                err += MSE::calculate(&truth[i], temp_input);

                // backward propagation
                let error = MSE::derivative(&truth[i], temp_input);
                temp_input = &error;
                for l in self.layers.iter_mut().rev() {
                    temp_out = l.backward_prop(temp_input, learning_rate);
                    temp_input = &temp_out;
                }
            }
            err /= sample_len as f64;
            println!("epoch {}/{}, error: {:.6}", epoch + 1, epochs, err);
        }
    }
}

pub struct SigmoidLayer {
    pub input: Array<f64>,
}

impl SigmoidLayer {
    pub fn new() -> Self {
        SigmoidLayer {
            input: Array::empty(0, 0),
        }
    }
}

impl Layer for SigmoidLayer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.input = Array::empty(input.len(), 1);
        self.input.data = input.to_vec();
        calculate(&input, Sigmoid::activation)
    }

    fn backward_prop(&mut self, error: &Vec<f64>, _learning_rate: f64) -> Vec<f64> {
        let mut vec = calculate(&self.input.data, Sigmoid::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        vec
    }
}

pub struct ReLULayer {
    pub input: Array<f64>,
}

impl ReLULayer {
    pub fn new() -> Self {
        ReLULayer {
            input: Array::empty(0, 0),
        }
    }
}

impl Layer for ReLULayer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.input = Array::empty(input.len(), 1);
        self.input.data = input.to_vec();
        calculate(&input, ReLU::activation)
    }

    fn backward_prop(&mut self, error: &Vec<f64>, _learning_rate: f64) -> Vec<f64> {
        let mut vec = calculate(&self.input.data, ReLU::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        vec
    }
}

pub struct TanHLayer {
    pub input: Array<f64>,
}

impl TanHLayer {
    pub fn new() -> Self {
        TanHLayer {
            input: Array::empty(0, 0),
        }
    }
}

impl Layer for TanHLayer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.input = Array::empty(input.len(), 1);
        self.input.data = input.to_vec();
        calculate(&input, TanH::activation)
    }

    fn backward_prop(&mut self, error: &Vec<f64>, _learning_rate: f64) -> Vec<f64> {
        let mut vec = calculate(&self.input.data, TanH::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        vec
    }
}

pub struct DenseLayer {
    pub input: Array<f64>,
    pub weights: Array<f64>,
    pub bias: Array<f64>,
}

impl DenseLayer {
    pub fn new(rows: usize, cols: usize) -> Self {
        DenseLayer {
            input: Array::empty(rows, 1),
            weights: Array::random(rows, cols),
            bias: Array::empty(1, cols),
        }
    }
}

impl Layer for DenseLayer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.input.data = input.to_vec();
        let rows = self.weights.rows;
        let cols = self.weights.cols;
        
        let mut vec = vec![0 as f64; cols];
        for i in 0..rows {
            for j in 0..cols {
                vec[j] += self.weights.data[i * cols + j] * self.input.data[i];
            }
        }
        for j in 0..cols {
            vec[j] += self.bias.data[j];
        }
        vec
    }

    fn backward_prop(&mut self, error: &Vec<f64>, learn_rate: f64) -> Vec<f64> {
        let error_array = Array {
            rows: 1,
            cols: self.weights.cols,
            data: error.to_vec(),
        };

        let input_error = error_array.dot(&self.weights.t());
        let weights_error = self.input.dot(&error_array);

        self.weights.add(weights_error * -learn_rate);
        self.bias.add(error_array * -learn_rate);

        input_error.to_vec()
    }
}
