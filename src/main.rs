mod utils;

use utils::{layer::{DenseLayer, TanHLayer}, model::Sequential, shape::Array};

use utils::{ dataset::MnistData };
use std::time::{Instant};

use crate::utils::layer::{ReLULayer, SigmoidLayer};

fn main() {
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
    model.add(SigmoidLayer::new());
    model.add(DenseLayer::new(100, 50));
    model.add(SigmoidLayer::new());
    model.add(DenseLayer::new(50, 10));
    model.add(SigmoidLayer::new());

    let timer = Instant::now();
    model.train(&train_image_data.data[0..2048].to_vec(), &train_label_data.data[0..2048].to_vec(), 50, 1, 0.1);
    println!("Training time: {} ms", timer.elapsed().as_millis());

    let res = model.predict(&test_image_data.data[0..4].to_vec());
    for i in 0..res.len() {
        println!("{:?}", &res[i]);
        println!("{:?}", &test_label_data.data[i]);
    }
}
