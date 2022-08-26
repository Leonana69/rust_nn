mod utils;

use utils::{layer::{DenseLayer, TanHLayer}, model::Sequential};

fn main() {
    let mut input_data: Vec<Vec<f64>> = Vec::default();
    let mut model = Sequential::new();
    let mut truth: Vec<Vec<f64>> = Vec::default();

    input_data.push(vec![0.0, 0.0]);
    input_data.push(vec![0.0, 1.0]);
    input_data.push(vec![1.0, 0.0]);
    input_data.push(vec![1.0, 1.0]);
    
    truth.push(vec![0.0]);
    truth.push(vec![1.0]);
    truth.push(vec![1.0]);
    truth.push(vec![0.0]);

    model.add(DenseLayer::new(2, 3));
    model.add(TanHLayer::new());
    model.add(DenseLayer::new(3, 1));
    model.add(TanHLayer::new());
    
    model.train(&input_data, &truth, 1000, 1, 0.1);

    let res = model.predict(&input_data);
    for i in res.iter() {
        println!("{:?}", i);
    }
}
