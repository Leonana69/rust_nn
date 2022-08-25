pub trait Loss {
    fn calculate(truth: &[f64], predict: &[f64]) -> f64;
    fn derivative(truth: &[f64], predict: &[f64]) -> Box<[f64]>;
}

pub struct MSE;

impl Loss for MSE {
    fn calculate(truth: &[f64], predict: &[f64]) -> f64 {
        let mut sq_diff = 0.0;
        for (yt, yh) in truth.into_iter().zip(predict.into_iter()) {
            sq_diff += (yt - yh).powf(2.0);
        }
        sq_diff / truth.len() as f64
    }
    fn derivative(truth: &[f64], predict: &[f64]) -> Box<[f64]> {
        let len = truth.len();
        let len2 = 2.0 / len as f64;
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push((predict[i] - truth[i]) * len2);
        }
        vec.into_boxed_slice()
    }
}