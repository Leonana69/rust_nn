use std::mem::replace;

use super::{ops::{ Sigmoid, ReLU, Operator, calculate, TanH, ReLU6 }, shape::Array, loss::MSE};
use crate::utils::loss::Loss;
pub trait Layer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64>;
    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>);
    fn config_shape(&mut self, prev_output_shape: &[usize]);
    fn update_parameters(&mut self, _delta_weights: &Array<f64>, _delta_bias: &Array<f64>) {}
    fn set_parameters(&mut self, _weights: &Array<f64>, _bias: &Array<f64>) {}
}

macro_rules! new_activation_layer {
    ($struct:ident, $type:ident) => {
        pub struct $struct {
            pub input: Array<f64>,
            pub input_shape: Box<[usize]>,
            pub output_shape: Box<[usize]>,
        }

        impl $struct {
            pub fn new() -> Self {
                $struct {
                    input: Array::empty(),
                    input_shape: Box::default(),
                    output_shape: Box::default(),
                }
            }
        }

        impl Layer for $struct {
            fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
                self.input = input.clone();
                calculate(input, $type::activation)
            }
        
            fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
                let mut input = Array::empty();
                input = replace(&mut self.input, input);
                let mut deriv = calculate(input, $type::derivative);
                for i in 0..error.sub_size[0] {
                    deriv.data[i] = deriv.data[i] * error.data[i];
                }
                (deriv, None, None)
            }
        
            fn config_shape(&mut self, prev_output_shape: &[usize]) {
                self.input_shape = prev_output_shape.into();
                self.output_shape = prev_output_shape.into();
            }
        }
    };
}

new_activation_layer!(SigmoidLayer, Sigmoid);
new_activation_layer!(ReLULayer, ReLU);
new_activation_layer!(ReLU6Layer, ReLU6);
new_activation_layer!(TanHLayer, TanH);

// pub struct SigmoidLayer {
//     pub input: Array<f64>,
//     pub input_shape: Box<[usize]>,
//     pub output_shape: Box<[usize]>,
// }

// impl SigmoidLayer {
//     pub fn new() -> Self {
//         SigmoidLayer {
//             input: Array::empty(),
//             input_shape: Box::default(),
//             output_shape: Box::default(),
//         }
//     }
// }

// impl Layer for SigmoidLayer {
//     fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
//         self.input = input.clone();
//         calculate(input, Sigmoid::activation)
//     }

//     fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
//         let mut input = Array::empty();
//         input = replace(&mut self.input, input);
//         let mut deriv = calculate(input, Sigmoid::derivative);
//         for i in 0..error.sub_size[0] {
//             deriv.data[i] = deriv.data[i] * error.data[i];
//         }
//         (deriv, None, None)
//     }

//     fn config_shape(&mut self, prev_output_shape: &[usize]) {
//         self.input_shape = prev_output_shape.into();
//         self.output_shape = prev_output_shape.into();
//     }
// }

// pub struct ReLULayer {
//     pub input: Array<f64>,
//     pub input_shape: Box<[usize]>,
//     pub output_shape: Box<[usize]>,
// }

// impl ReLULayer {
//     pub fn new() -> Self {
//         ReLULayer {
//             input: Array::empty(),
//             input_shape: Box::default(),
//             output_shape: Box::default(),
//         }
//     }
// }

// impl Layer for ReLULayer {
//     fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
//         self.input = input.clone();
//         calculate(input, ReLU::activation)
//     }

//     fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
//         let mut input = Array::empty();
//         input = replace(&mut self.input, input);
//         let mut deriv = calculate(input, ReLU::derivative);
//         for i in 0..error.sub_size[0] {
//             deriv.data[i] = deriv.data[i] * error.data[i];
//         }
//         (deriv, None, None)
//     }

//     fn config_shape(&mut self, prev_output_shape: &[usize]) {
//         self.input_shape = prev_output_shape.into();
//         self.output_shape = prev_output_shape.into();
//     }
// }

// pub struct ReLU6Layer {
//     pub input: Array<f64>,
//     pub input_shape: Box<[usize]>,
//     pub output_shape: Box<[usize]>,
// }

// impl ReLU6Layer {
//     pub fn new() -> Self {
//         ReLU6Layer {
//             input: Array::empty(),
//             input_shape: Box::default(),
//             output_shape: Box::default(),
//         }
//     }
// }

// impl Layer for ReLU6Layer {
//     fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
//         self.input = input.clone();
//         calculate(input, ReLU6::activation)
//     }

//     fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
//         let mut input = Array::empty();
//         input = replace(&mut self.input, input);
//         let mut deriv = calculate(input, ReLU6::derivative);
//         for i in 0..error.sub_size[0] {
//             deriv.data[i] = deriv.data[i] * error.data[i];
//         }
//         (deriv, None, None)
//     }

//     fn config_shape(&mut self, prev_output_shape: &[usize]) {
//         self.input_shape = prev_output_shape.into();
//         self.output_shape = prev_output_shape.into();
//     }
// }

// pub struct TanHLayer {
//     pub input: Array<f64>,
//     pub input_shape: Box<[usize]>,
//     pub output_shape: Box<[usize]>,
// }

// impl TanHLayer {
//     pub fn new() -> Self {
//         TanHLayer {
//             input: Array::empty(),
//             input_shape: Box::default(),
//             output_shape: Box::default(),
//         }
//     }
// }

// impl Layer for TanHLayer {
//     fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
//         self.input = input.clone();
//         calculate(input, TanH::activation)
//     }

//     fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
//         let mut input = Array::empty();
//         input = replace(&mut self.input, input);
//         let mut deriv = calculate(input, TanH::derivative);
//         for i in 0..error.sub_size[0] {
//             deriv.data[i] = deriv.data[i] * error.data[i];
//         }
//         (deriv, None, None)
//     }

//     fn config_shape(&mut self, prev_output_shape: &[usize]) {
//         self.input_shape = prev_output_shape.into();
//         self.output_shape = prev_output_shape.into();
//     }
// }

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
        self.input = input;
        let mut w = self.input.dot(&self.weights);
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

    fn config_shape(&mut self, prev_output_shape: &[usize]) {
        todo!()
    }
}

pub struct Conv2DLayer {
    pub kernel_size: usize,
    pub input: Array<f64>,
    pub weights: Array<f64>,
    pub bias: Array<f64>,
}

impl Conv2DLayer {
    pub fn new(output_channel: usize, kernel_size: usize, input_shape: &[usize]) -> Self {
        Conv2DLayer {
            kernel_size,
            input: Array::empty(),
            weights: Array::<f64>::random(&[input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, output_channel]),
            bias: Array::zeros(&[output_channel]),
        }
    }
}

impl Layer for Conv2DLayer {
    fn forward_prop(&mut self, input: Array<f64>) -> Array<f64> {
        let o_rows = self.weights.shape[0];
        let o_cols = self.weights.shape[1];
        let o_ch = self.weights.shape[2];
        let i_ch = input.shape[2];

        let k_size = self.kernel_size;

        let mut res: Array<f64> = Array::zeros(&[o_rows, o_cols, o_ch]);

        for i in 0..o_rows {
            for j in 0..o_cols {
                for k in 0..o_ch {
                    for ik in 0..i_ch {
                        for ki in 0..k_size {
                            for kj in 0..k_size {
                                res[&[i, j, k]] += self.weights[&[ki, kj, ik, k]] * input[&[i + ki, j + kj, ik]];
                            }
                        }
                    }
                    res[&[i, j, k]] += self.bias[&[k, 0]];
                }
            }
        }
        res
    }

    fn backward_prop(&mut self, error: Array<f64>) -> (Array<f64>, Option<Array<f64>>, Option<Array<f64>>) {
        // TODO!
        // dL/dX = full-conv(weights.rotate(180), error)
        // dL/dW = conv(input, error)
        // dL/dB = sum(error)
        (Array::empty(), None, None)
    }

    fn update_parameters(&mut self, delta_weights: &Array<f64>, delta_bias: &Array<f64>) {
        self.weights.add_m(delta_weights);
        self.bias.add_m(delta_bias);
    }

    fn set_parameters(&mut self, weights: &Array<f64>, bias: &Array<f64>) {
        self.weights = weights.clone();
        self.bias = bias.clone();
    }

    fn config_shape(&mut self, prev_output_shape: &[usize]) {
        todo!()
    }
}
