use std::f64::consts::E;
pub trait Operator {
    fn activation(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
    fn calculate(input: &[f64], op_cal: fn(f64) -> f64) -> Box<[f64]> {
        let len = input.len();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(op_cal(input[i]));
        }
        vec.into_boxed_slice()
    }
}
#[derive(Default)]
pub struct Sigmoid;

#[derive(Default)]
pub struct ReLU;

impl Operator for Sigmoid {
    fn activation(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }
    fn derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }
}

impl Operator for ReLU {
    fn activation(x: f64) -> f64 {
        x.max(0.0)
    }
    fn derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
