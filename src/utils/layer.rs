pub trait Layer {
    fn init(&mut self);
    
}

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
}