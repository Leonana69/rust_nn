mod utils;

use utils::{layer::{DenseLayer, TanHLayer}, model::Sequential, shape::Array};

use utils::{ dataset::MnistData };
use std::time::{Instant};
fn main() {
    // let rows = 784;
    // let cols = 1000;

    // let mut a2 = vec![0.0; rows * cols];
    // let mut b2 = vec![0.0; rows * cols];
    // let A2 = a2.as_mut_slice();
    // let B2 = b2.as_mut_slice();
    // let now2 = Instant::now();
    // for i in 0..rows * cols {
    //     A2[i] = A2[i] + B2[i];
    // }
    // println!("2: {}", now2.elapsed().as_micros());

    // let mut a1: Array<f64> = Array::empty(rows, cols);
    // let mut b1: Array<f64> = Array::empty(rows, cols);
    // let now1 = Instant::now();
    // a1.add(&b1);
    // println!("1: {}", now1.elapsed().as_micros());

    // let mut a3 = [0.0 as f64; 1200];
    // let mut b3 = [0.0 as f64; 1200];
    // let now3 = Instant::now();
    // for i in 0..1200 {
    //     a3[i] = a3[i] + b3[i];
    // }
    // println!("3: {}", now3.elapsed().as_micros());

    println!("Loading data...");
    let train_image_data = MnistData::new("./dataset/train-images.idx3-ubyte").unwrap();
    let train_label_data = MnistData::new("./dataset/train-labels.idx1-ubyte").unwrap();

    let test_image_data = MnistData::new("./dataset/t10k-images.idx3-ubyte").unwrap();
    let test_label_data = MnistData::new("./dataset/t10k-labels.idx1-ubyte").unwrap();

    println!("Loaded image [number, width, height]: {:?}", &train_image_data.sizes);
    println!("Loaded label [number]: {:?}", &train_label_data.sizes);

    println!("Start training...");
    let mut model = Sequential::new();
    model.add(DenseLayer::new(784, 100));
    model.add(TanHLayer::new());
    model.add(DenseLayer::new(100, 50));
    model.add(TanHLayer::new());
    model.add(DenseLayer::new(50, 10));
    model.add(TanHLayer::new());

    let timer = Instant::now();
    model.train(&train_image_data.data[0..256].to_vec(), &train_label_data.data[0..256].to_vec(), 5, 1, 0.1);
    println!("Training cost: {} ms", timer.elapsed().as_millis());

    let res = model.predict(&test_image_data.data[0..4].to_vec());
    for i in 0..res.len() {
        println!("{:?}", &res[i]);
        println!("{:?}", &test_label_data.data[i]);
    }
}
