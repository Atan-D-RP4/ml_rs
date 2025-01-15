use crate::matrix::Matrix;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Relu(f32),
    Tanh,
    Sin,
}

impl Activation {
    fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Relu(param) => {
                if x > 0.0 {
                    x
                } else {
                    x * param
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Sin => x.sin(),
        }
    }

    fn derivative(&self, y: f32) -> f32 {
        match self {
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Relu(param) => {
                if y >= 0.0 {
                    1.0
                } else {
                    *param
                }
            }
            Activation::Tanh => 1.0 - y * y,
            Activation::Sin => (y.asin()).cos(),
        }
    }
}

impl Matrix<f32> {
    pub fn apply_activation(&mut self, activation: Activation) {
        for element in &mut self.elements {
            *element = activation.forward(*element);
        }
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub architecture: Vec<usize>,
    pub weights: Vec<Matrix<f32>>,
    pub biases: Vec<Matrix<f32>>,
    pub activations: Vec<Matrix<f32>>,
    pub activation_fn: Activation,
}

impl NeuralNetwork {
    pub fn new(architecture: &[usize], activation: Activation) -> Self {
        assert!(architecture.len() >= 2);

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();

        // Initialize activations for each layer
        for &size in architecture {
            activations.push(Matrix::new(1, size));
        }

        // Initialize weights and biases between layers
        for i in 0..architecture.len() - 1 {
            weights.push(Matrix::new(architecture[i], architecture[i + 1]));
            biases.push(Matrix::new(1, architecture[i + 1]));
        }

        Self {
            architecture: architecture.to_vec(),
            weights,
            biases,
            activations,
            activation_fn: activation,
        }
    }

    pub fn randomize(&mut self, low: f32, high: f32) {
        for weight in &mut self.weights {
            weight.randomize(low, high);
        }
        for bias in &mut self.biases {
            bias.randomize(low, high);
        }
    }

    pub fn predict(&mut self, input: &[f32]) -> &Matrix<f32> {
        self.forward(&input);
        self.activations.last().unwrap()
    }

    pub fn forward(&mut self, input: &[f32]) {
        assert_eq!(input.len(), self.architecture[0]);

        // Set input layer
        self.activations[0].elements.copy_from_slice(input);

        // Forward propagation
        for i in 0..self.weights.len() {
            let activation = self.activations[i].dot(&self.weights[i]);
            self.activations[i + 1] = activation;
            self.activations[i + 1].add(&self.biases[i]);
            self.activations[i + 1].apply_activation(self.activation_fn);
        }
    }

    pub fn cost(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        let mut total_cost = 0.0;

        for (input, target) in training_data {
            self.forward(input);
            let output = &self.activations.last().unwrap().elements;

            for (out, target) in output.iter().zip(target.iter()) {
                let diff = out - target;
                total_cost += diff * diff;
            }
        }

        total_cost / training_data.len() as f32
    }

    pub fn backpropagation(&mut self, training_data: &[(Vec<f32>, Vec<f32>)], learning_rate: f32) {
        let batch_size = training_data.len();
        let mut weight_gradients: Vec<Matrix<f32>> = self
            .weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();
        let mut bias_gradients: Vec<Matrix<f32>> = self
            .biases
            .iter()
            .map(|b| Matrix::new(b.rows, b.cols))
            .collect();

        for (input, target) in training_data {
            self.forward(input);

            // Calculate output layer error
            let mut deltas = vec![Matrix::new(1, self.architecture.last().unwrap().clone())];
            let output_layer = self.activations.last().unwrap();

            for j in 0..output_layer.cols {
                let output = output_layer[(0, j)];
                let error = output - target[j];
                deltas[0][(0, j)] = error * self.activation_fn.derivative(output);
            }

            // Backpropagate error
            for layer in (0..self.weights.len()).rev() {
                let delta = &deltas[0];

                // Calculate gradients
                for i in 0..self.weights[layer].rows {
                    for j in 0..self.weights[layer].cols {
                        weight_gradients[layer][(i, j)] +=
                            self.activations[layer][(0, i)] * delta[(0, j)];
                    }
                }

                for j in 0..self.biases[layer].cols {
                    bias_gradients[layer][(0, j)] += delta[(0, j)];
                }

                if layer > 0 {
                    let mut new_delta = Matrix::new(1, self.architecture[layer]);
                    for i in 0..new_delta.cols {
                        let mut sum = 0.0;
                        for j in 0..delta.cols {
                            sum += self.weights[layer][(i, j)] * delta[(0, j)];
                        }
                        new_delta[(0, i)] = sum
                            * self
                                .activation_fn
                                .derivative(self.activations[layer][(0, i)]);
                    }
                    deltas[0] = new_delta;
                }
            }
        }

        // Update weights and biases
        let learning_rate = learning_rate / batch_size as f32;
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].elements.len() {
                self.weights[i].elements[j] -= learning_rate * weight_gradients[i].elements[j];
            }
            for j in 0..self.biases[i].elements.len() {
                self.biases[i].elements[j] -= learning_rate * bias_gradients[i].elements[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_neural_network() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1], Activation::Sigmoid);
        nn.randomize(-1.0, 1.0);

        let training_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let initial_cost = nn.cost(&training_data);

        // Train for a few epochs
        for _ in 0..1000 {
            nn.backpropagation(&training_data, 0.1);
        }

        let final_cost = nn.cost(&training_data);
        assert!(final_cost < initial_cost);
    }
}
