mod dataset;
mod error;
mod network;

#[cfg(test)]
mod test;

pub use dataset::DataSet;
pub use error::NNError;
pub use network::{Activation, Architecture, NeuralNetwork};
