mod utils;

use utils::{layer::{DenseLayer, SigmoidLayer, Conv2DLayer}, model::Sequential};
use utils::{dataset::MnistData};
use std::time::{Instant};

use crate::utils::layer::InputLayer;

fn main() {
    // test_mnist();
    test_face();
}

fn test_face() {
    let mut model = Sequential::new();
    model.add(InputLayer::new(&[64, 64, 3]));
    model.add(Conv2DLayer::new(32, 3));
    model.add(SigmoidLayer::new());
    model.compile();
}

fn test_mnist() {
    // load Mnist training and test dataset
    println!("Loading data...");
    let train_image_data = MnistData::new("./dataset/train-images.idx3-ubyte").unwrap();
    let train_label_data = MnistData::new("./dataset/train-labels.idx1-ubyte").unwrap();

    let test_image_data = MnistData::new("./dataset/t10k-images.idx3-ubyte").unwrap();
    let test_label_data = MnistData::new("./dataset/t10k-labels.idx1-ubyte").unwrap();

    println!("Loaded image [number, width, height]: {:?}", &train_image_data.sizes);
    println!("Loaded label [number]: {:?}", &train_label_data.sizes);

    println!("Start training...");
    let mut model = Sequential::new();
    model.add(InputLayer::new(&[1, 784]));
    model.add(DenseLayer::new(100));
    model.add(SigmoidLayer::new());
    model.add(DenseLayer::new(50));
    model.add(SigmoidLayer::new());
    model.add(DenseLayer::new(10));
    model.add(SigmoidLayer::new());

    model.compile();
    let timer = Instant::now();
    model.train(&train_image_data.data[0..2048].to_vec(), &train_label_data.data[0..2048].to_vec(), &[1, 784], 50, 1, 0.2);
    println!("Training time: {} s", timer.elapsed().as_secs());

    let res = model.predict(&test_image_data.data[0..4], &[1, 784]);
    for i in 0..res.len() {
        println!("{:?}", &res[i]);
        println!("{:?}", &test_label_data.data[i]);
    }
}