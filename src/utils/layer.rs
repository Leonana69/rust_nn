use super::{ops::{ Sigmoid, ReLU, Operator, calculate, TanH, ReLU6 }, shape::Array, loss::MSE};
use crate::utils::loss::Loss;
pub trait Layer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64>;
    fn backward_prop(&mut self, error: &Vec<f64>) -> (Vec<f64>, Option<Array<f64>>, Option<Array<f64>>);
    fn update_parameters(&mut self, _delta_weights: &Array<f64>, _delta_bias: &Array<f64>) {}
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

    fn backward_prop(&mut self, error: &Vec<f64>) -> (Vec<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut vec = calculate(&self.input.data, Sigmoid::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        (vec, None, None)
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

    fn backward_prop(&mut self, error: &Vec<f64>) -> (Vec<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut vec = calculate(&self.input.data, ReLU::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        (vec, None, None)
    }
}

pub struct ReLU6Layer {
    pub input: Array<f64>,
}

impl ReLU6Layer {
    pub fn new() -> Self {
        ReLU6Layer {
            input: Array::empty(0, 0),
        }
    }
}

impl Layer for ReLU6Layer {
    fn forward_prop(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.input = Array::empty(input.len(), 1);
        self.input.data = input.to_vec();
        calculate(&input, ReLU6::activation)
    }

    fn backward_prop(&mut self, error: &Vec<f64>) -> (Vec<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut vec = calculate(&self.input.data, ReLU6::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        (vec, None, None)
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

    fn backward_prop(&mut self, error: &Vec<f64>) -> (Vec<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut vec = calculate(&self.input.data, TanH::derivative);
        for i in 0..error.len() {
            vec[i] = vec[i] * error[i];
        }
        (vec, None, None)
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
            weights: Array::<f64>::random(rows, cols),
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

    fn backward_prop(&mut self, error: &Vec<f64>) -> (Vec<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let error_array = Array {
            rows: 1,
            cols: self.weights.cols,
            data: error.to_vec(),
        };

        let input_error = error_array.dot(&self.weights.t());
        let weights_error = self.input.dot(&error_array);

        // self.weights.add(weights_error * -learn_rate);
        // self.bias.add(error_array * -learn_rate);
        (input_error.to_vec(), Some(weights_error), Some(error_array))
    }

    fn update_parameters(&mut self, delta_weights: &Array<f64>, delta_bias: &Array<f64>) {
        self.weights.add(delta_weights);
        self.bias.add(delta_bias);
    }
}
