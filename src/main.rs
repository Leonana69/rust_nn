mod utils;

use utils::{layer::{DenseLayer, TanHLayer}, model::Sequential};

use utils::{ dataset::MnistData };

fn main() {
    // let mut input_data: Vec<Vec<f64>> = Vec::default();
    // let mut model = Sequential::new();
    // let mut truth: Vec<Vec<f64>> = Vec::default();

    // input_data.push(vec![0.0, 0.0]);
    // input_data.push(vec![0.0, 1.0]);
    // input_data.push(vec![1.0, 0.0]);
    // input_data.push(vec![1.0, 1.0]);
    
    // truth.push(vec![0.0]);
    // truth.push(vec![1.0]);
    // truth.push(vec![1.0]);
    // truth.push(vec![0.0]);

    // model.add(DenseLayer::new(2, 3));
    // model.add(TanHLayer::new());
    // model.add(DenseLayer::new(3, 1));
    // model.add(TanHLayer::new());
    
    // model.train(&input_data, &truth, 1000, 1, 0.1);

    // let res = model.predict(&input_data);
    // for i in res.iter() {
    //     println!("{:?}", i);
    // }
    println!("Loading data...");
    let train_image_data = MnistData::new("./dataset/train-images.idx3-ubyte").unwrap();
    let train_label_data = MnistData::new("./dataset/train-labels.idx1-ubyte").unwrap();

    let test_image_data = MnistData::new("./dataset/t10k-images.idx3-ubyte").unwrap();
    let test_label_data = MnistData::new("./dataset/t10k-labels.idx1-ubyte").unwrap();

        println!("{:?}", &test_label_data.data[0..4]);
        return;

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

    model.train(&train_image_data.data[0..512].to_vec(), &train_label_data.data[0..512].to_vec(), 4, 1, 0.1);
    
    let res = model.predict(&test_image_data.data[0..4].to_vec());
    for i in res.iter() {
        println!("{:?}", i);
    }
}
