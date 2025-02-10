use crate::matrix::Matrix;
use rand::distributions::Distribution;
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum NNError {
    ArchitectureError {
        msg: String,
        details: Option<String>,
    },
    InputError {
        msg: String,
        expected_size: usize,
        actual_size: usize,
    },
    DataSetError {
        msg: String,
        stride: usize,
        total_columns: usize,
    },
    MatrixError {
        msg: String,
        operation: String,
    },
    TrainingError {
        msg: String,
        cost: Option<f32>,
    },
}

impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NNError::ArchitectureError { msg, details } => {
                write!(f, "Architecture Error: {}", msg)?;
                if let Some(detail) = details {
                    write!(f, " ({})", detail)?;
                }
                Ok(())
            }
            NNError::InputError {
                msg,
                expected_size,
                actual_size,
            } => {
                write!(
                    f,
                    "Input Error: {} (expected {}, got {})",
                    msg, expected_size, actual_size
                )
            }
            NNError::DataSetError {
                msg,
                stride,
                total_columns,
            } => {
                write!(
                    f,
                    "DataSet Error: {} (stride: {}, total columns: {})",
                    msg, stride, total_columns
                )
            }
            NNError::MatrixError { msg, operation } => {
                write!(f, "Matrix Operation Error: {} during {}", msg, operation)
            }
            NNError::TrainingError { msg, cost } => {
                write!(f, "Training Error: {}", msg)?;
                if let Some(c) = cost {
                    write!(f, " (current cost: {})", c)?;
                }
                Ok(())
            }
        }
    }
}

impl Error for NNError {}

#[derive(Debug, Default)]
pub struct DataSet {
    pub data: Matrix<f32>,
    pub stride: usize,
}

impl DataSet {
    pub fn new(data: Matrix<f32>, stride: usize) -> Result<Self, NNError> {
        if stride >= data.cols {
            return Err(NNError::DataSetError {
                msg: "Stride cannot be greater than the total columns".to_string(),
                stride,
                total_columns: data.cols,
            });
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

    pub fn derivative(&self, y: f32) -> f32 {
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

pub struct Architecture {
    pub inputs: usize,
    pub outputs: usize,
    pub layers: &'static [usize],
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub architecture: Vec<usize>,
    pub weights: Vec<Matrix<f32>>,
    pub biases: Vec<Matrix<f32>>,
    pub activation_fn: Activation,
    activations: Vec<Matrix<f32>>,
}

impl NeuralNetwork {
    pub fn validate_architecture(&self) -> Result<(), NNError> {
        if self.architecture.is_empty() {
            return Err(NNError::ArchitectureError {
                msg: "Empty architecture".to_string(),
                details: None,
            });
        }

        if self.weights.len() != self.architecture.len() - 1 {
            return Err(NNError::ArchitectureError {
                msg: "Inconsistent weights and architecture".to_string(),
                details: Some(format!(
                    "Weights count: {}, Architecture layers: {}",
                    self.weights.len(),
                    self.architecture.len() - 1
                )),
            });
        }

        Ok(())
    }

    pub fn new(arch: Architecture, activation: Activation) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut architecture = Vec::new();
        architecture.push(arch.inputs);
        arch.layers.iter().for_each(|neuron_count| {
            architecture.push(*neuron_count);
        });
        architecture.push(arch.outputs);

        let acts_count = architecture.len();
        for i in 0..(acts_count - 1) {
            weights.push(Matrix::new(architecture[i], architecture[i + 1]));
            biases.push(Matrix::new(1, architecture[i + 1]));
        }

        // Pre-allocate activations with correct dimensions
        let mut activations = Vec::with_capacity(acts_count);
        for size in &architecture {
            activations.push(Matrix::new(1, *size));
        }

        Self {
            architecture,
            weights,
            biases,
            activation_fn: activation,
            activations,
        }
    }

    pub fn init_parameters(&mut self, input_size: usize) -> Result<(), NNError> {
        self.validate_architecture()?;
        if input_size != self.architecture[0] {
            return Err(NNError::InputError {
                msg: "Input size does not match the size of the input layer".to_string(),
                expected_size: self.architecture[0],
                actual_size: input_size,
            });
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

    pub fn predict(&mut self, input: &Matrix<f32>) -> Result<Matrix<f32>, NNError> {
        self.forward(input)?;
        Ok(self.activations.last().unwrap().clone())
    }

    fn forward(&mut self, input: &Matrix<f32>) -> Result<(), NNError> {
        // Validate input dimensions
        if input.cols != self.architecture[0] || input.rows != 1 {
            return Err(NNError::InputError {
                msg: "Invalid input dimensions".to_string(),
                expected_size: self.architecture[0],
                actual_size: input.cols,
            });
        }

        // Set input layer
        self.activations[0] = input.clone();

        // Forward propagation
        for i in 0..self.weights.len() {
            // Compute dot product and store in the next layer's activations
            self.activations[i + 1] =
                self.activations[i]
                    .dot(&self.weights[i])
                    .map_err(|e| NNError::MatrixError {
                        msg: e.to_string(),
                        operation: "dot product".to_string(),
                    })?;

            // Add biases
            self.activations[i + 1]
                .add(&self.biases[i])
                .map_err(|e| NNError::MatrixError {
                    msg: e.to_string(),
                    operation: "dot product".to_string(),
                })?;

            // Apply activation function
            self.activations[i + 1].apply_activation(self.activation_fn);
        }
        Ok(())
    }

    pub fn cost(&mut self, dataset: &DataSet) -> Result<f32, NNError> {
        let inputs = dataset.inputs();
        let targets = dataset.targets();
        let mut total_cost = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            self.forward(input)?;
            let output = &self.activations.last().unwrap().elements;

            for (out, target_val) in output.iter().zip(target.elements.iter()) {
                let diff = out - target_val;
                total_cost += diff * diff;
            }
        }

        Ok(total_cost / inputs.len() as f32)
    }

    fn clip_gradients(gradient: &mut Matrix<f32>, threshold: f32) {
        gradient.apply_fn(|x| x.clamp(-threshold, threshold));
    }

    fn learn(
        &mut self,
        w_gradients: &mut Vec<Matrix<f32>>,
        b_gradients: &mut Vec<Matrix<f32>>,
        rate: f32,
        batch_size: usize,
    ) -> Result<(), NNError> {
        for gradient in &mut *w_gradients {
            if gradient
                .elements
                .iter()
                .any(|&x| x.is_nan() || x.is_infinite())
            {
                return Err(NNError::TrainingError {
                    msg: "Gradient explosion detected".to_string(),
                    cost: None,
                });
            }
        }

        for gradient in &mut *b_gradients {
            if gradient
                .elements
                .iter()
                .any(|&x| x.is_nan() || x.is_infinite())
            {
                return Err(NNError::TrainingError {
                    msg: "Gradient explosion detected".to_string(),
                    cost: None,
                });
            }
        }

        let learning_rate = rate / batch_size as f32;
        for i in 0..self.weights.len() {
            Self::clip_gradients(&mut w_gradients[i], 1.0);
            Self::clip_gradients(&mut b_gradients[i], 1.0);
            w_gradients[i].apply_fn(|x| x * learning_rate);
            b_gradients[i].apply_fn(|x| x * learning_rate);
            self.weights[i].sub(&w_gradients[i]).unwrap();
            self.biases[i].sub(&b_gradients[i]).unwrap();
        }
        Ok(())
    }

    pub fn backpropagation(
        &mut self,
        weight_gradients: &mut Vec<Matrix<f32>>,
        bias_gradients: &mut Vec<Matrix<f32>>,
        dataset: &DataSet,
        learning_rate: f32,
    ) -> Result<(), NNError> {
        let inputs = dataset.inputs();
        let targets = dataset.targets();
        let batch_size = inputs.len();

        weight_gradients.iter_mut().for_each(|g| g.fill(0.0));
        bias_gradients.iter_mut().for_each(|g| g.fill(0.0));

        let mut deltas = self
            .architecture
            .iter()
            .map(|&size| Matrix::<f32>::new(1, size))
            .collect::<Vec<Matrix<f32>>>();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            self.forward(input)?;

            // Output layer error
            let output_layer_idx = self.activations.len() - 1;
            let output_layer = &mut self.activations[output_layer_idx];

            // Compute output layer delta using element-wise operations
            for j in 0..output_layer.cols {
                let output = output_layer[(0, j)];
                let error = output - target[(0, j)];
                deltas[output_layer_idx][(0, j)] = error * self.activation_fn.derivative(output);
            }

            // Backpropagate Error
            for layer in (0..self.weights.len()).rev() {
                let delta = &deltas[layer + 1];

                // Weight gradients
                for i in 0..self.weights[layer].rows {
                    for j in 0..self.weights[layer].cols {
                        weight_gradients[layer][(i, j)] +=
                            self.activations[layer][(0, i)] * delta[(0, j)];
                    }
                }

                // Bias gradients
                for j in 0..self.biases[layer].cols {
                    bias_gradients[layer][(0, j)] += delta[(0, j)];
                }

                if layer > 0 {
                    let mut new_delta = Matrix::new(1, self.architecture[layer]);

                    for i in 0..new_delta.cols {
                        let mut sum = 0.0;
                        for j in 0..delta.cols {
                            sum += delta[(0, j)] * self.weights[layer][(i, j)];
                        }
                        new_delta[(0, i)] = sum
                            * self
                                .activation_fn
                                .derivative(self.activations[layer][(0, i)]);
                    }
                    deltas[layer] = new_delta;
                }
            }
        }

        self.learn(weight_gradients, bias_gradients, learning_rate, batch_size)?;
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
    fn test_bin_op_nn() -> Result<(), Box<dyn std::error::Error>> {
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
        let arch = Architecture {
            inputs: dataset.stride,
            layers: &[3],
            outputs: dataset.data.cols - dataset.stride,
        }; // NOTE: [2, 3, 1] - 1 hidden layers
        let mut nn = NeuralNetwork::new(arch, Activation::Sigmoid);
        nn.init_parameters(dataset.stride)?;
        let mut bias_gradients: Vec<Matrix<f32>> = nn
            .biases
            .iter()
            .map(|b| Matrix::new(b.rows, b.cols))
            .collect();
        let mut weight_gradients: Vec<Matrix<f32>> = nn
            .weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();

        println!("{}", nn);

        let initial_cost = nn.cost(&dataset)?;
        println!("Initial Cost: {}", initial_cost);

        let start = std::time::Instant::now();
        // Train for a few epochs
        for i in 0..30000 {
            nn.backpropagation(&mut weight_gradients, &mut bias_gradients, &dataset, 1e0)?;
        }

        let final_cost = nn.cost(&dataset)?;
        println!("Final Cost: {}", final_cost);
        println!("Time Elapsed: {:?}", start.elapsed());
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
        assert!(false);
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

        let arch = Architecture {
            inputs: dataset.stride,
            layers: &[6],
            outputs: dataset.data.cols - dataset.stride,
        };
        let mut nn = NeuralNetwork::new(arch, Activation::Sigmoid);
        println!("Initialising parameters...");
        nn.init_parameters(dataset.stride)?;
        let mut bias_gradients: Vec<Matrix<f32>> = nn
            .biases
            .iter()
            .map(|b| Matrix::new(b.rows, b.cols))
            .collect();
        let mut weight_gradients: Vec<Matrix<f32>> = nn
            .weights
            .iter()
            .map(|w| Matrix::new(w.rows, w.cols))
            .collect();

        println!("{}", nn);

        let initial_cost = nn.cost(&dataset)?;
        println!("Initial Cost: {}", initial_cost);

        let start = std::time::Instant::now();
        // Train for a few epochs
        for i in 0..50000 {
            nn.backpropagation(&mut weight_gradients, &mut bias_gradients, &dataset, 1e0)?;
        }
        println!("Time Elapsed: {:?}", start.elapsed());

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
        assert!(false);
        Ok(())
    }
}
