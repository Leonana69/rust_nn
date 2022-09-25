use super::shape::Array;

pub trait Loss {
    fn calculate(truth: &Vec<f64>, predict: &Array<f64>) -> f64;
    fn derivative(truth: &Vec<f64>, predict: Array<f64>) -> Array<f64>;
}

#[derive(Default)]
pub struct MSE;

impl Loss for MSE {
    fn calculate(truth: &Vec<f64>, predict: &Array<f64>) -> f64 {
        let mut sq_diff = 0.0;
        for (yt, yh) in truth.into_iter().zip(predict.data.into_iter()) {
            sq_diff += (yt - yh).powf(2.0);
        }
        sq_diff / truth.len() as f64
    }
    fn derivative(truth: &Vec<f64>, mut predict: Array<f64>) -> Array<f64> {
        let len = truth.len();
        let len2 = 2.0 / len as f64;
        
        for i in 0..len {
            predict.data[i] = (predict.data[i] - truth[i]) * len2;
        }
        predict
    }
}