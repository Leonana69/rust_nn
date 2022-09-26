mod utils;

use utils::{layer::{DenseLayer, SigmoidLayer, Conv2DLayer}, model::Sequential, shape::Array};
use utils::{dataset::MnistData};
use std::{time::{Instant}, fs};

use crate::utils::layer::{InputLayer, ReLULayer};

fn main() {
    test_mnist();
    // test_face();
}

#[allow(dead_code)]
fn test_face() {
    // load bias
    let content = fs::read("./resource/bias.txt").unwrap();
    let content = std::str::from_utf8(&content).unwrap();
    let sp = content.split(",");
    let mut bias = Vec::<f64>::new();
    for s in sp {
        bias.push(s.parse::<f64>().unwrap());
    }
    let bias = Array::<f64>::with(&[32], &bias);

    // load weights
    let content = fs::read("./resource/weights.txt").unwrap();
    let content = std::str::from_utf8(&content).unwrap();
    let sp = content.split(",");
    let mut weights = Vec::<f64>::new();
    for s in sp {
        weights.push(s.parse::<f64>().unwrap());
    }
    let weights = Array::<f64>::with(&[3, 3, 3, 32], &weights);

    // load image
    let content = fs::read("./resource/test_image.txt").unwrap();
    let content = std::str::from_utf8(&content).unwrap();
    let sp = content.split(",");
    let mut input_img = Vec::<f64>::new();
    for s in sp {
        input_img.push(s.parse::<f64>().unwrap());
    }
    

    let mut model = Sequential::new();
    model.add(InputLayer::new(&[64, 64, 3]));
    model.add(Conv2DLayer::new(32, 3));
    model.add(ReLULayer::new());
    model.compile();
    model.layers[1].set_parameters(weights, bias);

    let res = model.predict(&[input_img]);

    let res_str: Vec<String> = res[0].iter().map(|n| n.to_string()).collect();
    fs::write("./resource/temp.txt", res_str.join(", ").as_bytes()).unwrap();
}

#[allow(dead_code)]
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
    model.train(&train_image_data.data[0..2048].to_vec(), &train_label_data.data[0..2048].to_vec(), 50, 1, 0.2);
    println!("Training time: {} s", timer.elapsed().as_secs());

    let res = model.predict(&test_image_data.data[0..4]);
    for i in 0..res.len() {
        println!("Case {i}:");
        println!("guess: {:.2?}", &res[i]);
        println!("truth: {:.2?}", &test_label_data.data[i]);
    }
}