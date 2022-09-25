use std::mem::replace;

use super::{ops::{ Sigmoid, ReLU, Operator, calculate, TanH, ReLU6 }, shape::Array, loss::MSE};
use crate::utils::loss::Loss;
pub trait Layer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64>;
    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>);
    fn update_parameters(&mut self, _delta_weights: &Array<f64>, _delta_bias: &Array<f64>) {}
    fn set_parameters(&mut self, _weights: &Array<f64>, _bias: &Array<f64>) {}
}

pub struct SigmoidLayer {
    pub input: Array<f64>,
}

impl SigmoidLayer {
    pub fn new() -> Self {
        SigmoidLayer {
            input: Array::empty(),
        }
    }
}

impl Layer for SigmoidLayer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        self.input = input.clone();
        calculate(input, Sigmoid::activation)
    }

    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut input = Array::empty();
        input = replace(&mut self.input, input);
        let mut deriv = calculate(input, Sigmoid::derivative);
        for i in 0..error.sub_size[0] {
            deriv.data[i] = deriv.data[i] * error.data[i];
        }
        (deriv, None, None)
    }
}

pub struct ReLULayer {
    pub input: Array<f64>,
}

impl ReLULayer {
    pub fn new() -> Self {
        ReLULayer {
            input: Array::empty(),
        }
    }
}

impl Layer for ReLULayer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        self.input = input.clone();
        calculate(input, ReLU::activation)
    }

    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut input = Array::empty();
        input = replace(&mut self.input, input);
        let mut deriv = calculate(input, ReLU::derivative);
        for i in 0..error.sub_size[0] {
            deriv.data[i] = deriv.data[i] * error.data[i];
        }
        (deriv, None, None)
    }
}

pub struct ReLU6Layer {
    pub input: Array<f64>,
}

impl ReLU6Layer {
    pub fn new() -> Self {
        ReLU6Layer {
            input: Array::empty(),
        }
    }
}

impl Layer for ReLU6Layer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        self.input = input.clone();
        calculate(input, ReLU6::activation)
    }

    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut input = Array::empty();
        input = replace(&mut self.input, input);
        let mut deriv = calculate(input, ReLU6::derivative);
        for i in 0..error.sub_size[0] {
            deriv.data[i] = deriv.data[i] * error.data[i];
        }
        (deriv, None, None)
    }
}

pub struct TanHLayer {
    pub input: Array<f64>,
}

impl TanHLayer {
    pub fn new() -> Self {
        TanHLayer {
            input: Array::empty(),
        }
    }
}

impl Layer for TanHLayer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        self.input = input.clone();
        calculate(input, TanH::activation)
    }

    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let mut input = Array::empty();
        input = replace(&mut self.input, input);
        let mut deriv = calculate(input, TanH::derivative);
        for i in 0..error.sub_size[0] {
            deriv.data[i] = deriv.data[i] * error.data[i];
        }
        (deriv, None, None)
    }
}

pub struct DenseLayer {
    pub input: Array<f64>,
    pub weights: Array<f64>,
    pub bias: Array<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        DenseLayer {
            input: Array::zeros(&[input_size]),
            weights: Array::<f64>::random(&[input_size, output_size]),
            bias: Array::zeros(&[1, output_size]),
        }
    }
}

impl Layer for DenseLayer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        self.input = input.clone();
        let mut w = input.dot(&self.weights);
        w.add_m(&self.bias);
        w
    }

    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        let input_error = error.dot(&self.weights.t());
        let weights_error = self.input.t().dot(&error);

        (input_error, Some(weights_error), Some(error))
    }

    fn update_parameters(&mut self, delta_weights: &Array<f64>, delta_bias: &Array<f64>) {
        self.weights.add_m(delta_weights);
        self.bias.add_m(delta_bias);
    }
}

pub struct Conv2DLayer {
    pub input: Array<f64>,
    pub weights: Array<f64>,
    pub bias: Array<f64>,
}

// impl Conv2DLayer {
//     pub fn new(output_channel: usize, kernel_size: usize, input_shape: &[usize]) -> Self {
//         Conv2DLayer {
//             input: Array::zeros(&[0]),
//             weights: Array::<f64>::random(&[input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, output_channel]),
//             bias: Array::zeros(&[1, output_channel]),
//         }
//     }
// }

// impl Layer for Conv2DLayer {
//     fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        
//     }

//     fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
//         let error_array = Array::with(&[1, self.weights.shape[1]], &error.to_vec());

//         let input_error = error_array.dot(&self.weights.t());
//         let weights_error = self.input.dot(&error_array);

//         (input_error.to_vec(), Some(weights_error), Some(error_array))
//     }

//     fn update_parameters(&mut self, delta_weights: &Array<f64>, delta_bias: &Array<f64>) {
//         self.weights.add_m(delta_weights);
//         self.bias.add_m(delta_bias);
//     }

//     fn set_parameters(&mut self, weights: &Array<f64>, bias: &Array<f64>) {
//         self.weights = weights.clone();
//         self.bias = bias.clone();
//     }
// }

// fn Conv2D(w: Array<f64>, b: Array<f64>, input: Array<f64>) -> Array<f64> {
//     let o_rows = input.shape[0] - w.shape[0] + 1;
//     let o_cols = input.shape[1] - w.shape[1] + 1;
//     let o_ch = b.shape[0];
//     let i_ch = input.shape[2];

//     let k_rows = w.shape[0];
//     let k_cols = w.shape[1];

//     let mut res: Array<f64> = Array::zeros(&[o_rows, o_cols, o_ch]);

//     for i in 0..o_rows {
//         for j in 0..o_cols {
//             for k in 0..o_ch {
//                 for ik in 0..i_ch {
//                     for ki in 0..k_rows {
//                         for kj in 0..k_cols {
//                             res[&[i, j, k]] += w[&[ki, kj, ik, k]] * input[&[i + ki, j + kj, ik]];
//                         }
//                     }
//                 }
//                 res[&[i, j, k]] += b[&[k, 0]];
//             }
//         }
//     }

//     res
// }
