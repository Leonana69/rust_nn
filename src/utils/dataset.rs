use std::{fs::{File, self}, io::Read};

pub struct MnistData {
    pub sizes: Vec<i32>,
    pub data: Vec<Vec<f64>>,
}

impl MnistData {
    pub fn new(fileName: &str) -> Result<MnistData, std::io::Error> {
        let metadata = fs::metadata(fileName).expect("Unable to read metadata");

        let mut buffer = vec![0; metadata.len() as usize];
        let mut file = File::open(fileName).expect("Unable to open file.");
        file.read(&mut buffer).expect("Buffer overflow.");

        let magic_number = i32::from_be_bytes(buffer[0..4].try_into().unwrap());

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<Vec<f64>> = Vec::default();

        match magic_number {
            2049 => {
                sizes.push(i32::from_be_bytes(buffer[4..8].try_into().unwrap()));
                data = buffer[8..].to_vec().iter().map(|&x| {
                    let mut v = vec![0.0 as f64; 10]; v[x as usize] = 1.0;
                    v
                }).collect();
                Ok(MnistData { sizes, data })
            }
            2051 => {
                sizes.push(i32::from_be_bytes(buffer[4..8].try_into().unwrap()));
                sizes.push(i32::from_be_bytes(buffer[8..12].try_into().unwrap()));
                sizes.push(i32::from_be_bytes(buffer[12..16].try_into().unwrap()));
                let image_size = (sizes[1] * sizes[2]) as usize;
                for i in 0..sizes[0] as usize {
                    let start: usize = 16 + image_size * i;
                    let end: usize = start + image_size;
                    data.push(buffer[start..end].to_vec().iter().map(|&x| x as f64 / 255.0).collect());
                }

                Ok(MnistData { sizes, data })
            }
            _ => panic!(),
        }
    }
}