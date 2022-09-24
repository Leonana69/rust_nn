use std::{f64::consts::E};
pub trait Operator {
    fn activation(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}
#[derive(Default)]
pub struct Sigmoid;

#[derive(Default)]
pub struct ReLU;

#[derive(Default)]
pub struct ReLU6;

pub struct TanH;

impl Operator for Sigmoid {
    fn activation(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }
    fn derivative(x: f64) -> f64 {
        let sx = Self::activation(x);
        sx * (1.0 - sx)
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

impl Operator for ReLU6 {
    fn activation(x: f64) -> f64 {
        x.max(0.0).min(6.0)
    }
    fn derivative(x: f64) -> f64 {
        if x > 0.0 && x < 6.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl Operator for TanH {
    fn activation(x: f64) -> f64 {
        x.tanh()
    }
    fn derivative(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}

pub fn calculate(input: &Vec<f64>, op_cal: fn(f64) -> f64) -> Vec<f64> {
    let len = input.len();
    let mut vec = Vec::with_capacity(len);
    for i in 0..len {
        vec.push(op_cal(input[i]));
    }
    vec
}
