use ndarray::{Array1, Array2, Axis};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
    Sin,
}

const RELU_PARAM: f32 = 0.01;

impl Activation {
    fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Relu => if x > 0.0 { x } else { x * RELU_PARAM },
            Activation::Tanh => x.tanh(),
            Activation::Sin => x.sin(),
        }
    }

    fn derivative(&self, y: f32) -> f32 {
        match self {
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Relu => if y >= 0.0 { 1.0 } else { RELU_PARAM },
            Activation::Tanh => 1.0 - y * y,
            Activation::Sin => (y.asin()).cos(),
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.gen_range(-1.0..1.0)
        });

        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut output = self.weights.dot(input);
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

        let mut layers = Vec::new();
        for window in architecture.windows(2) {
            layers.push(Layer::new(window[0], window[1], activation));
        }

        let activations = architecture.iter()
            .map(|&size| Array1::zeros(size))
            .collect();

        Self {
            layers,
            activations,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> &Array1<f32> {
        assert_eq!(input.len(), self.activations[0].len(),
            "Input size does not match network architecture");

        // Set input layer
        self.activations[0].assign(&Array1::from_vec(input.to_vec()));

        // Forward propagation through each layer
        for i in 0..self.layers.len() {
            let input = self.activations[i].clone();
            self.activations[i + 1] = self.layers[i].forward(&input);
        }

        // Return reference to output layer
        &self.activations[self.activations.len() - 1]
    }

    pub fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)],
                 learning_rate: f32, batch_size: usize) -> f32 {
        let mut total_loss = 0.0;
        let num_samples = training_data.len();

        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch = &training_data[batch_start..batch_end];

            // Accumulate gradients
            let (weight_gradients, bias_gradients) = self.compute_gradients(batch);

            // Update weights and biases
            let batch_lr = learning_rate / batch.len() as f32;
            for ((layer, weight_grad), bias_grad) in self.layers.iter_mut()
                .zip(weight_gradients)
                .zip(bias_gradients) {
                layer.weights -= &(weight_grad * batch_lr);
                layer.biases -= &(bias_grad * batch_lr);
            }

            // Compute loss for this batch
            total_loss += self.compute_loss(batch);
        }

        total_loss / num_samples as f32
    }

    fn compute_gradients(&mut self, batch: &[(Vec<f32>, Vec<f32>)])
        -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {

        let mut weight_gradients: Vec<Array2<f32>> = self.layers.iter()
            .map(|layer| Array2::zeros(layer.weights.raw_dim()))
            .collect();

        let mut bias_gradients: Vec<Array1<f32>> = self.layers.iter()
            .map(|layer| Array1::zeros(layer.biases.raw_dim()))
            .collect();

        for (input, target) in batch {
            // Forward pass
            self.forward(input);

            // Backward pass
            let mut delta = {
                let output = self.activations.last().unwrap();
                let target = Array1::from_vec(target.clone());
                let mut delta = output - &target;
                delta *= &output.mapv(|y|
                    self.layers.last().unwrap().activation.derivative(y));
                delta
            };

            // Propagate error backwards through the network
            for layer_idx in (0..self.layers.len()).rev() {
                let layer = &self.layers[layer_idx];
                let input = &self.activations[layer_idx];

                // Compute gradients for this layer
                let weight_grad = delta.clone().insert_axis(Axis(1)) * input.clone().insert_axis(Axis(0));
                weight_gradients[layer_idx] += &weight_grad;
                bias_gradients[layer_idx] += &delta;

                if layer_idx > 0 {
                    // Compute delta for next layer back
                    let next_delta = layer.weights.t().dot(&delta);
                    delta = next_delta * &self.activations[layer_idx]
                        .mapv(|y| self.layers[layer_idx-1].activation.derivative(y));
                }
            }
        }

        (weight_gradients, bias_gradients)
    }

    pub fn compute_loss(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        let mut loss = 0.0;

        for (input, target) in batch {
            let output = self.forward(input);
            let target = Array1::from_vec(target.clone());
            loss += output.iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>();
        }

        loss / (2.0 * batch.len() as f32)
    }

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

        // Train for several epochs
        for _ in 0..1000 {
            nn.train(&training_data, 0.1, 4);
        }

        let final_loss = nn.compute_loss(&training_data);
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(2, 1, Activation::Sigmoid);
        let input = Array1::from_vec(vec![1.0, 1.0]);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 1);
    }
}
