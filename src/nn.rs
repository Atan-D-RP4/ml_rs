use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// Previous Activation enum and impl remain the same
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
    Sin,
}

const RELU_PARAM: f32 = 0.01;

impl Activation {
    #[inline]
    fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Relu => x.max(x * RELU_PARAM),
            Activation::Tanh => x.tanh(),
            Activation::Sin => x.sin(),
        }
    }

    #[inline]
    fn derivative(&self, y: f32) -> f32 {
        match self {
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Relu => if y >= 0.0 { 1.0 } else { RELU_PARAM },
            Activation::Tanh => 1.0 - y * y,
            Activation::Sin => (y.asin()).cos(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = thread_rng();
        let scale = 1.0 / (input_size as f32).sqrt();
        let dist = Uniform::new(-scale, scale);

        let weights = Array2::from_shape_fn((output_size, input_size), |_| dist.sample(&mut rng));
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
        }
    }

    #[inline]
    pub fn forward(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut output = self.weights.dot(&input);
        output += &self.biases;
        output.mapv_inplace(|x| self.activation.forward(x));
        output
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    activations: Vec<Array1<f32>>,
}

impl NeuralNetwork {
    pub fn new(architecture: &[usize], activation: Activation) -> Self {
        assert!(architecture.len() >= 2, "Network must have at least input and output layers");

        let layers: Vec<_> = architecture.windows(2)
            .map(|window| Layer::new(window[0], window[1], activation))
            .collect();

        let activations = architecture.iter()
            .map(|&size| Array1::zeros(size))
            .collect();

        Self {
            layers,
            activations,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Array1<f32> {
        self.activations[0].assign(&Array1::from_vec(input.to_vec()));

        for i in 0..self.layers.len() {
            let input = self.activations[i].view();
            self.activations[i + 1] = self.layers[i].forward(input);
        }

        self.activations.last().unwrap().clone()
    }

    fn process_batch(layers: &[Layer], batch: &[(Vec<f32>, Vec<f32>)])
        -> (Vec<(Array2<f32>, Array1<f32>)>, f32) {

        let mut gradients: Vec<(Array2<f32>, Array1<f32>)> = layers.iter()
            .map(|layer| (
                Array2::zeros(layer.weights.raw_dim()),
                Array1::zeros(layer.biases.raw_dim())
            ))
            .collect();

        let mut total_loss = 0.0;
        let mut activations: Vec<Array1<f32>> = vec![Array1::zeros(0); layers.len() + 1];

        for (input, target) in batch {
            // Forward pass
            activations[0] = Array1::from_vec(input.clone());

            for i in 0..layers.len() {
                let input = activations[i].view();
                activations[i + 1] = layers[i].forward(input);
            }

            let output = activations.last().unwrap();
            let target = Array1::from_vec(target.clone());

            // Compute loss
            total_loss += output.iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>();

            // Backward pass
            let mut delta = {
                let output = activations.last().unwrap();
                let mut delta = output - &target;
                delta *= &output.mapv(|y| layers.last().unwrap().activation.derivative(y));
                delta
            };

            // Propagate error backwards
            for layer_idx in (0..layers.len()).rev() {
                let layer = &layers[layer_idx];
                let input = &activations[layer_idx];

                // Accumulate gradients
                let weight_grad = delta.clone().insert_axis(Axis(1)) * input.clone().insert_axis(Axis(0));
                gradients[layer_idx].0 += &weight_grad;
                gradients[layer_idx].1 += &delta;

                if layer_idx > 0 {
                    let next_delta = layer.weights.t().dot(&delta);
                    delta = next_delta * &activations[layer_idx]
                        .mapv(|y| layers[layer_idx-1].activation.derivative(y));
                }
            }
        }

        (gradients, total_loss / (2.0 * batch.len() as f32))
    }

    pub fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)],
                 learning_rate: f32, batch_size: usize) -> f32 {
        let num_samples = training_data.len();
        let layers = Arc::new(self.layers.clone());
        let gradients_mutex = Arc::new(Mutex::new(vec![(Array2::zeros((0, 0)), Array1::zeros(0)); self.layers.len()]));
        let total_loss = Arc::new(Mutex::new(0.0f32));

        // Process batches in parallel
        (0..num_samples).step_by(batch_size).par_bridge().for_each(|batch_start| {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch = &training_data[batch_start..batch_end];

            // Compute gradients and loss for this batch
            let (batch_gradients, batch_loss) = Self::process_batch(&layers, batch);

            // Accumulate gradients and loss under lock
            let mut total_gradients = gradients_mutex.lock().unwrap();
            let mut total_batch_loss = total_loss.lock().unwrap();

            for i in 0..batch_gradients.len() {
                if total_gradients[i].0.is_empty() {
                    total_gradients[i] = (
                        Array2::zeros(batch_gradients[i].0.raw_dim()),
                        Array1::zeros(batch_gradients[i].1.raw_dim())
                    );
                }
                total_gradients[i].0 += &batch_gradients[i].0;
                total_gradients[i].1 += &batch_gradients[i].1;
            }
            *total_batch_loss += batch_loss;
        });

        // Apply accumulated gradients
        let gradients = gradients_mutex.lock().unwrap();
        let batch_lr = learning_rate / num_samples as f32;

        for (layer, (weight_grad, bias_grad)) in self.layers.iter_mut().zip(gradients.iter()) {
            layer.weights -= &(weight_grad * batch_lr);
            layer.biases -= &(bias_grad * batch_lr);
        }

        let x = *total_loss.lock().unwrap();
        x
    }

    pub fn compute_loss(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        let mut total_loss = 0.0;

        for (input, target) in batch {
            let output = self.forward(input);
            let target = Array1::from_vec(target.clone());
            total_loss += output.iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>();
        }

        total_loss / (2.0 * batch.len() as f32)
    }

    #[inline]
    pub fn predict(&mut self, input: &[f32]) -> Vec<f32> {
        self.forward(input).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor() {
        let mut nn = NeuralNetwork::new(&[2, 4, 1], Activation::Sigmoid);
        let training_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let initial_loss = nn.compute_loss(&training_data);

        for _ in 0..1000 {
            nn.train(&training_data, 0.1, 4);
        }

        let final_loss = nn.compute_loss(&training_data);
        assert!(final_loss < initial_loss);

        // Test predictions
        for (input, expected) in &training_data {
            let prediction = nn.predict(input);
            assert!((prediction[0] - expected[0]).abs() < 0.1);
        }
    }
}
