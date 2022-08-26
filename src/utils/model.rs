use crate::utils::loss::{MSE, Loss};

use super::layer::Layer;

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

    pub fn train(&mut self, input: &Vec<Vec<f64>>, truth: &Vec<Vec<f64>>, epoches: usize, batch_size: usize, learning_rate: f64) {
        let sample_len = input.len();

        assert!(sample_len > 0 && truth.len() == sample_len && batch_size > 0 && learning_rate > 0.0);

        for epoch in 0..epoches {
            let mut err = 0.0; // error on all samples
            let iters: usize = sample_len / batch_size;

            for i in 0..iters + 1 {
                let mut error_sum = vec![0.0; truth[0].len()];
                let mut bs = batch_size;
                if i == iters {
                    bs = sample_len % batch_size;
                }

                for b in 0..bs {
                    let mut temp_input = &input[i * batch_size + b];
                    let mut temp_output: Vec<f64>;
                    for l in self.layers.iter_mut() {
                        temp_output = l.forward_prop(temp_input);
                        temp_input = &temp_output;
                    }
                    err += MSE::calculate(&truth[i], temp_input);
                    error_sum = error_sum.iter().zip(temp_input.iter()).map(|(&e, &o)| e + o).collect();
                }

                if bs > 0 {
                    // backward propagation
                    let loss = MSE::derivative(&truth[i], &error_sum);
                    let mut back_input = &loss;
                    let mut back_output: Vec<f64>;
                    for l in self.layers.iter_mut().rev() {
                        back_output = l.backward_prop(back_input, learning_rate);
                        back_input = &back_output;
                    }
                }
            }

            // for i in 0..sample_len {
            //     let mut temp_input = &input[i];
            //     let mut temp_out: Vec<f64>;
            //     for l in self.layers.iter_mut() {
            //         temp_out = l.forward_prop(temp_input);
            //         temp_input = &temp_out;
            //     }
            //     err += MSE::calculate(&truth[i], temp_input);

            //     // backward propagation
            //     let error = MSE::derivative(&truth[i], temp_input);
            //     temp_input = &error;
            //     for l in self.layers.iter_mut().rev() {
            //         temp_out = l.backward_prop(temp_input, learning_rate);
            //         temp_input = &temp_out;
            //     }
            // }

            err /= sample_len as f64;
            println!("epoch {}/{}, error: {:.6}", epoch + 1, epoches, err);
        }
    }
}