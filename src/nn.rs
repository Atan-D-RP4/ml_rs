use crate::matrix::{Matrix, MatrixError};
use rand::distributions::Distribution;

#[derive(Debug, Default)]
pub struct DataSet {
    pub data: Matrix<f32>,
    stride: usize,
}

impl DataSet {
    pub fn new(data: Matrix<f32>, stride: usize) -> Result<Self, &'static str> {
        if stride >= data.cols {
            return Err("Stride must be less than the number of columns in the data matrix");
        }
        Ok(Self { data, stride })
    }

    pub fn inputs(&self) -> Vec<Matrix<f32>> {
        let mut inputs = Vec::new();
        for i in 0..self.data.rows {
            let mut inp_row = Vec::new();
            for j in 0..self.stride {
                inp_row.push(self.data[(i, j)]);
            }
            inputs.push(Matrix::from_vec2d(vec![inp_row]).unwrap());
        }
        inputs
    }

    pub fn inputs_as_matrix(&self) -> Matrix<f32> {
        let inputs = self.inputs();
        inputs
            .iter()
            .fold(Matrix::new(0, inputs[0].cols), |mut acc, x| {
                acc.vstack(x).expect("Matrix stacking failed");
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
            targets.push(Matrix::from_vec2d(vec![target_row]).unwrap());
        }
        targets
    }

    pub fn targets_as_matrix(&self) -> Matrix<f32> {
        let targets = self.targets();
        targets
            .iter()
            .fold(Matrix::new(0, targets[0].cols), |mut acc, x| {
                acc.vstack(x).expect("Matrix stacking failed");
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
    pub activation_fn: Activation,
}

impl NeuralNetwork {
    pub fn new(architecture: &[usize], activation: Activation) -> Self {
        assert!(architecture.len() >= 2);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..architecture.len() - 1 {
            weights.push(Matrix::new(architecture[i], architecture[i + 1]));
            biases.push(Matrix::new(1, architecture[i + 1]));
        }

        Self {
            architecture: architecture.to_vec(),
            weights,
            biases,
            activation_fn: activation,
        }
    }

    pub fn init_parameters(&mut self, input_size: usize) -> Result<(), &str> {
        if input_size != self.architecture[0] {
            return Err("Input size does not match the size of the input layer");
        }
        let scale = 1.0 / (input_size as f32).sqrt();
        let dist = rand::distributions::Uniform::new(-scale, scale);

        for i in 0..self.architecture.len() - 1 {
            self.weights[i].apply_fn(|_| dist.sample(&mut rand::thread_rng()));
            self.biases[i].randomize(scale, -scale);
        }
        Ok(())
    }

    pub fn randomize(&mut self, low: f32, high: f32) {
        for weight in &mut self.weights {
            weight.randomize(low, high);
        }
        for bias in &mut self.biases {
            bias.randomize(low, high);
        }
    }

    pub fn predict(&mut self, input: &Matrix<f32>) -> Result<Matrix<f32>, MatrixError> {
        assert_eq!(input.cols, self.architecture[0]);
        let activations = self.forward(input)?;
        Ok(activations.last().unwrap().clone())
    }

    fn forward(&self, input: &Matrix<f32>) -> Result<Vec<Matrix<f32>>, MatrixError> {
        // Set input layer
        let mut activations = Vec::new();
        for size in self.architecture.iter() {
            activations.push(Matrix::new(1, *size));
        }
        activations[0] = input.clone();

        // Forward propagation
        for i in 0..self.weights.len() {
            let activation = activations[i].dot(&self.weights[i])?;
            activations[i + 1] = activation;
            activations[i + 1].add(&self.biases[i])?;
            activations[i + 1].apply_activation(self.activation_fn);
        }
        Ok(activations)
    }

    pub fn cost(&self, dataset: &DataSet) -> Result<f32, MatrixError> {
        let inputs = dataset.inputs();
        let targets = dataset.targets();
        let mut total_cost = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let activations = self.forward(input)?;
            let output = &activations.last().unwrap().elements;

            for (out, target_val) in output.iter().zip(target.elements.iter()) {
                let diff = out - target_val;
                total_cost += diff * diff;
            }
        }

        Ok(total_cost / inputs.len() as f32)
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

    pub fn backpropagation(
        &mut self,
        dataset: &DataSet,
        learning_rate: f32,
    ) -> Result<(), MatrixError> {
        let samples = dataset.inputs();
        let targets = dataset.targets();
        let batch_size = samples.len();

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

        for (input, target) in samples.iter().zip(targets.iter()) {
            let activations = self.forward(input)?;

            // Calculate output layer error
            let mut deltas = vec![Matrix::new(1, self.architecture.last().unwrap().clone())];
            let output_layer = activations.last().unwrap();

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
                            activations[layer][(0, i)] * delta[(0, j)];
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
                        new_delta[(0, i)] =
                            sum * self.activation_fn.derivative(activations[layer][(0, i)]);
                    }
                    deltas[0] = new_delta;
                }
            }
        }

        self.learn(weight_gradients, bias_gradients, learning_rate, batch_size);
        Ok(())
    }
}

impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NeuralNetwork {{\n")?;
        write!(f, "Architecture: {:?}\n", self.architecture)?;
        write!(f, "Weights: [")?;
        for weight in &self.weights {
            write!(f, "{}]\n", weight)?;
        }
        write!(f, "]\n")?;
        write!(f, "Biases: [\n")?;
        for bias in &self.biases {
            write!(f, "{}\n", bias)?;
        }
        write!(f, "]\n")?;
        write!(f, "}}")
    }
}

impl std::fmt::Display for DataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DataSet {{\n")?;
        // [[inputs], [targets]] based on stride
        self.data.elements.iter().enumerate().for_each(|(i, val)| {
            if i % self.data.cols == 0 {
                write!(f, "  [").unwrap();
            }
            write!(f, "{}, ", val).unwrap();
            if (i + 1) % self.data.cols == self.stride {
                write!(f, "|").unwrap();
            }
            if (i + 1) % self.data.cols == 0 {
                write!(f, "]\n").unwrap();
            }
        });

        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_op_neural_network() -> Result<(), Box<dyn std::error::Error>> {
        // Create XOR training data using DataSet
        let xor_data = Matrix::from_vec2d(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ])
        .unwrap();
        let dataset = DataSet::new(xor_data, 2)?; // 2 input columns, 1 target column

        // NOTE: arch -> [inputs, [neurons in each layer]..., outputs]
        // NOTE: arch can be [2, 2, 1, 2] or [2, 5, 5, 7, 2]
        let arch = [dataset.stride, 3, dataset.data.cols - dataset.stride]; // NOTE: [2, 3, 1] - 1 hidden layers
        let mut nn = NeuralNetwork::new(&arch, Activation::Sigmoid);
        // nn.init_parameters(dataset.stride)?;
        nn.init_parameters(dataset.stride)?;
        println!("{}", nn);

        let initial_cost = nn.cost(&dataset)?;

        // Train for a few epochs
        for i in 0..40000 {
            nn.backpropagation(&dataset, 1e-1)?;
            println!("{i}: Cost: {}", nn.cost(&dataset)?);
        }

        let final_cost = nn.cost(&dataset)?;
        println!("Final Cost: {}", final_cost);
        assert!(final_cost < initial_cost);

        // Test predictions
        (0..2).for_each(|i| {
            (0..2).for_each(|j| {
                let input = Matrix::from_vec2d(vec![vec![i as f32, j as f32]]).unwrap();
                let output = nn.predict(&input).expect("Prediction failed");
                println!("{} XOR {} = {}", i, j, output[(0, 0)]);
                assert_eq!(output[(0, 0)].round(), (i ^ j) as f32);
            });
        });
        Ok(())
    }

    #[test]
    fn test_binsum_nn() -> Result<(), Box<dyn std::error::Error>> {
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
        ])
        .unwrap();
        let dataset = DataSet::new(adder_data, 3)?; // 3 input columns, 2 target columns
        println!("{dataset}");

        // Test predictions
        let (inputs, targets) = (
            dataset.inputs_as_matrix().to_vec2d(),
            dataset.targets_as_matrix().to_vec2d(),
        );

        let arch = [dataset.stride, 6, dataset.data.cols - dataset.stride];
        let mut nn = NeuralNetwork::new(&arch, Activation::Sigmoid);
        println!("Initialising parameters...");
        nn.init_parameters(dataset.stride)?;
        println!("{}", nn);

        let initial_cost = nn.cost(&dataset)?;
        println!("Initial Cost: {}", initial_cost);

        // Train for a few epochs
        for i in 0..40000 {
            nn.backpropagation(&dataset, 1e-1)?;
            println!("{i}: Cost: {}", nn.cost(&dataset)?);
        }

        let final_cost = nn.cost(&dataset)?;
        println!("Final Cost: {}", final_cost);
        assert!(final_cost < initial_cost);

        for i in 0..inputs.len() {
            let input = Matrix::from_vec2d(vec![inputs[i].clone()]).unwrap();
            let output = nn.predict(&input)?;
            println!(
                "{} {} {} -> {} {}",
                inputs[i][0],
                inputs[i][1],
                inputs[i][2],
                output[(0, 0)],
                output[(0, 1)]
            );
            assert_eq!(output[(0, 0)].round(), targets[i][0]);
            assert_eq!(output[(0, 1)].round(), targets[i][1]);
        }
        Ok(())
    }
}
