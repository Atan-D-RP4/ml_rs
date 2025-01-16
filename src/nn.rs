use crate::matrix::Matrix;

#[derive(Debug)]
pub struct DataSet {
    pub data: Matrix<f32>,
    stride: usize,
}

impl DataSet {
    pub fn new(data: Matrix<f32>, stride: usize) -> Self {
        if stride >= data.cols {
            panic!("Stride must be less than the number of columns in the data matrix");
        }
        Self { data, stride }
    }

    pub fn inputs(&self) -> Vec<Matrix<f32>> {
        let mut inputs = Vec::new();
        for i in 0..self.data.rows {
            let mut inp_row = Vec::new();
            for j in 0..self.stride {
                inp_row.push(self.data[(i, j)]);
            }
            inputs.push(Matrix::from_vec2d(vec![inp_row]));
        }
        inputs
    }

    pub fn inputs_as_matrix(&self) -> Matrix<f32> {
        let inputs = self.inputs();
        inputs.iter().fold(Matrix::new(0, inputs[0].cols), |mut acc, x| {
            acc.vstack(x);
            acc
        })
    }

    pub fn targets(&self) -> Vec<Matrix<f32>> {
        let mut targets = Vec::new();
        for i in 0..self.data.rows {
            let mut target_row = Vec::new();
            for j in self.stride..self.data.cols {
                target_row.push(self.data[(i, j)]);
            }
            targets.push(Matrix::from_vec2d(vec![target_row]));
        }
        targets
    }

    pub fn targets_as_matrix(&self) -> Matrix<f32> {
        let targets = self.targets();
        targets.iter().fold(Matrix::new(0, targets[0].cols), |mut acc, x| {
            acc.vstack(x);
            acc
        })
    }
}

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
            Activation::Relu(param) => if x > 0.0 { x } else { x * param },
            Activation::Tanh => x.tanh(),
            Activation::Sin => x.sin(),
        }
    }

    fn derivative(&self, y: f32) -> f32 {
        match self {
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Relu(param) => if y >= 0.0 { 1.0 } else { *param },
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

        for &size in architecture {
            activations.push(Matrix::new(1, size));
        }

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

    pub fn predict(&mut self, input: &Matrix<f32>) -> &Matrix<f32> {
        assert_eq!(input.cols, self.architecture[0]);
        self.forward(input);
        self.activations.last().unwrap()
    }

    fn forward(&mut self, input: &Matrix<f32>) {
        // Set input layer
        self.activations[0] = input.clone();

        // Forward propagation
        for i in 0..self.weights.len() {
            let activation = self.activations[i].dot(&self.weights[i]);
            self.activations[i + 1] = activation;
            self.activations[i + 1].add(&self.biases[i]);
            self.activations[i + 1].apply_activation(self.activation_fn);
        }
    }

    pub fn cost(&mut self, dataset: &DataSet) -> f32 {
        let inputs = dataset.inputs();
        let targets = dataset.targets();
        let mut total_cost = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            self.forward(input);
            let output = &self.activations.last().unwrap().elements;

            for (out, target_val) in output.iter().zip(target.elements.iter()) {
                let diff = out - target_val;
                total_cost += diff * diff;
            }
        }

        total_cost / inputs.len() as f32
    }

    fn learn(
        &mut self,
        w_gradients: Vec<Matrix<f32>>,
        b_gradients: Vec<Matrix<f32>>,
        rate: f32,
        batch_size: usize,
    ) {
        let learning_rate = rate / batch_size as f32;
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].elements.len() {
                self.weights[i].elements[j] -= learning_rate * w_gradients[i].elements[j];
            }
            for j in 0..self.biases[i].elements.len() {
                self.biases[i].elements[j] -= learning_rate * b_gradients[i].elements[j];
            }
        }
    }

    pub fn backpropagation(&mut self, dataset: &DataSet, learning_rate: f32) {
        let inputs = dataset.inputs();
        let targets = dataset.targets();
        assert_eq!(inputs.len(), self.architecture[0]);
        assert_eq!(targets.len(), self.architecture[1]);
        let batch_size = inputs.len();

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

        for (input, target) in inputs.iter().zip(targets.iter()) {
            self.forward(input);

            // Calculate output layer error
            let mut deltas = vec![Matrix::new(1, self.architecture.last().unwrap().clone())];
            let output_layer = self.activations.last().unwrap();

            for j in 0..output_layer.cols {
                let output = output_layer[(0, j)];
                let error = output - target[(0, j)];
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
                        new_delta[(0, i)] = sum * self.activation_fn.derivative(self.activations[layer][(0, i)]);
                    }
                    deltas[0] = new_delta;
                }
            }
        }

        self.learn(weight_gradients, bias_gradients, learning_rate, batch_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_op_neural_network() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1], Activation::Sigmoid);
        nn.randomize(-1.0, 1.0);

        // Create XOR training data using DataSet
        let xor_data = Matrix::from_vec2d(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]);
        let dataset = DataSet::new(xor_data, 2); // 2 input columns, 1 target column

        let initial_cost = nn.cost(&dataset);

        // Train for a few epochs
        for i in 0..40000 {
            nn.backpropagation(&dataset, 1e-1);
            println!("{i}: Cost: {}", nn.cost(&dataset));
        }

        let final_cost = nn.cost(&dataset);
        assert!(final_cost < initial_cost);

        // Test predictions
        (0..2).for_each(|i| {
            (0..2).for_each(|j| {
                let input = Matrix::from_vec2d(vec![vec![i as f32, j as f32]]);
                let output = nn.predict(&input);
                println!("{} XOR {} = {}", i, j, output[(0, 0)]);
                assert_eq!(output[(0, 0)].round(), (i ^ j) as f32);
            });
        });
    }

    #[test]
    fn test_binsum_nn() {
        let mut nn = NeuralNetwork::new(&[3, 3, 2], Activation::Sigmoid);
        nn.randomize(-1.0, 1.0);
        println!("{:?}", nn);

        // Create XOR training data using DataSet
        let adder_data = Matrix::from_vec2d(vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ]);
        let dataset = DataSet::new(adder_data, 3); // 3 input columns, 2 target columns
        println!("Inputs:\n{}", dataset.inputs_as_matrix());
        println!("Targets:\n{}", dataset.targets_as_matrix());

        let initial_cost = nn.cost(&dataset);

        // Train for a few epochs
        for i in 0..40000 {
            nn.backpropagation(&dataset, 1e-1);
            println!("{i}: Cost: {}", nn.cost(&dataset));
        }

        let final_cost = nn.cost(&dataset);
        assert!(final_cost < initial_cost);

        // Test predictions
        let (inputs, targets) = (dataset.inputs_as_matrix().to_vec2d(), dataset.targets_as_matrix().to_vec2d());
        for i in 0..inputs.len() {
            let input = Matrix::from_vec2d(vec![inputs[i].clone()]);
            let output = nn.predict(&input);
            println!("{} {} {} -> {} {}", inputs[i][0], inputs[i][1], inputs[i][2], output[(0, 0)], output[(0, 1)]);
            assert_eq!(output[(0, 0)].round(), targets[i][0]);
            assert_eq!(output[(0, 1)].round(), targets[i][1]);
        }
        assert!(false)
    }
}
