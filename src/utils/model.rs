use crate::utils::{loss::{MSE, Loss}, shape::Array};
use super::layer::Layer;

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
                let mut bs = batch_size;
                // for leftover
                if i == iters {
                    bs = sample_len % batch_size;
                }

                if bs > 0 {
                    let mut vec_delta_weights: Vec<Option<Array<f64>>> = Vec::default();
                    let mut vec_delta_bias: Vec<Option<Array<f64>>> = Vec::default();
                    let layer_len = self.layers.len();
                    for b in 0..bs {
                        let mut layer_input = &input[i * batch_size + b];
                        let mut layer_output: Vec<f64> = Vec::new();
                        
                        for l in self.layers.iter_mut() {
                            layer_output = l.forward_prop(layer_input);
                            layer_input = &layer_output;
                        }
                        err += MSE::calculate(&truth[i], &layer_output);
                        
                        // backward propagation
                        let loss = MSE::derivative(&truth[i], &layer_output);
                        let mut back_input = &loss;
                        let mut back_output: Vec<f64>;
                        let mut delta_weights: Option<Array<f64>>;
                        let mut delta_bias: Option<Array<f64>>;

                        
                        for l in 0..layer_len {
                            (back_output, delta_weights, delta_bias) = self.layers[layer_len - 1 - l].backward_prop(back_input);

                            if b == 0 {
                                vec_delta_weights.push(delta_weights);
                                vec_delta_bias.push(delta_bias);
                            } else {
                                if let Some(w) = delta_weights {
                                    let vl = vec_delta_weights[l].as_mut().unwrap();
                                    vl.add(&w);
                                }
                                if let Some(b) = delta_bias {
                                    let vl = vec_delta_bias[l].as_mut().unwrap();
                                    vl.add(&b);
                                }
                            }
                            
                            back_input = &back_output;
                        }
                    }

                    for l in 0..layer_len {
                        println!("{:?}", vec_delta_weights[l]);
                        if let Some(_) = &vec_delta_weights[l] {
                            self.layers[layer_len - 1 - l].update_parameters(
                                vec_delta_weights[l].as_mut().unwrap().mul(-learning_rate),
                                vec_delta_bias[l].as_mut().unwrap().mul(-learning_rate)
                            );
                        }
                    }
                }
            }

            err /= sample_len as f64;
            println!("epoch {}/{}, error: {:.6}", epoch + 1, epoches, err);
        }
    }
}