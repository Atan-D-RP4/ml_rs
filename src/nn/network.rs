use crate::{
    matrix::Matrix,
    nn::{dataset::DataSet, error::NNError},
};
use rand::distributions::Distribution;

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
    pub fn apply_activation(&mut self, activation: Activation) -> &mut Self {
        for element in &mut self.elements {
            *element = activation.forward(*element);
        }
        self
    }
}

#[derive(Debug)]
pub struct Architecture {
    pub inputs: usize,
    pub layers: &'static [usize],
    pub outputs: usize,
    pub layer_count: usize,
}

impl Architecture {
    pub fn new() -> Self {
        Self {
            inputs: 0,
            layers: &[],
            outputs: 0,
            layer_count: 0,
        }
    }

    pub fn with(inputs: usize, layers: &'static [usize], outputs: usize) -> Self {
        Self {
            inputs,
            layers,
            outputs,
            layer_count: layers.len() + 2,
        }
    }

    pub fn validate(&self) -> Result<(), NNError> {
        if self.inputs == 0 {
            return Err(NNError::ArchitectureError {
                msg: "Input layer size is zero".to_string(),
                details: Some("Input layer size must be greater than zero".to_string()),
            });
        }
        if self.outputs == 0 {
            return Err(NNError::ArchitectureError {
                msg: "Output layer size is zero".to_string(),
                details: Some("Output layer size must be greater than zero".to_string()),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub arch: Architecture,
    pub weights: Vec<Matrix<f32>>,
    pub biases: Vec<Matrix<f32>>,
    pub activation_fn: Activation,
    activations: Vec<Matrix<f32>>,
}

impl NeuralNetwork {
    pub fn new(arch: Architecture, activation_fn: Activation) -> Result<Self, NNError> {
        arch.validate()?;
        let mut weights: Vec<Matrix<f32>> = Vec::new();
        let mut biases: Vec<Matrix<f32>> = Vec::new();

        // Initialize weights and biases
        weights.push(Matrix::new(arch.inputs, arch.layers[0]));
        biases.push(Matrix::new(1, arch.layers[0]));
        for i in 0..arch.layers.len() - 1 {
            weights.push(Matrix::new(arch.layers[i], arch.layers[i + 1]));
            biases.push(Matrix::new(1, arch.layers[i + 1]));
        }
        weights.push(Matrix::new(arch.layers[arch.layers.len() - 1], arch.outputs));
        biases.push(Matrix::new(1, arch.outputs));

        // Pre-allocate activations with correct dimensions
        let mut activations = Vec::with_capacity(arch.layer_count);
        activations.push(Matrix::new(1, arch.inputs));
        arch.layers.iter().for_each(|size| {
            activations.push(Matrix::new(1, *size));
        });
        activations.push(Matrix::new(1, arch.outputs));

        Ok(Self {
            arch,
            weights,
            biases,
            activation_fn,
            activations,
        })
    }

    pub fn init_parameters(&mut self, input_size: usize) -> Result<(), NNError> {
        if input_size != self.arch.inputs {
            return Err(NNError::InputError {
                msg: "Input size does not match the size of the input layer".to_string(),
                expected_size: self.arch.inputs,
                actual_size: input_size,
            });
        }
        let scale = 1.0 / (input_size as f32).sqrt();
        let dist = rand::distributions::Uniform::new(-scale, scale);

        for i in 0..self.arch.layer_count - 1 {
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
        if input.cols != self.arch.inputs || input.rows != 1 {
            return Err(NNError::InputError {
                msg: "Invalid input dimensions".to_string(),
                expected_size: self.arch.inputs,
                actual_size: input.cols,
            });
        }

        // Set input layer
        self.activations[0] = input.clone();

        // Forward propagation
        for i in 0..self.weights.len() {
            // Compute dot product and store in the next layer's activations
            self.activations[i + 1] = self.activations[i]
                .dot(&self.weights[i])? // Dot product with weights
                .add(&self.biases[i])? // Add bias
                .apply_activation(self.activation_fn) // Apply activation function
                .to_owned();
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

        // Pre-allocate vectors for better performance
        let mut deltas = self.activations.iter().map(|a| Matrix::new(1, a.cols)).collect::<Vec<Matrix<f32>>>();

        // Initialize gradients to zero
        for gradient in weight_gradients.iter_mut().chain(bias_gradients.iter_mut()) {
            gradient.fill(0.0);
        }

        let mut current_cost = 0.0;
        let mut max_gradient_norm: f32 = 0.0;

        for (batch_idx, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
            // Forward pass with cached activations
            self.forward(input)?;

            // Compute output layer error and gradients
            let output_layer_idx = self.activations.len() - 1;
            let ouput = &self.activations[output_layer_idx];
            let delta = &mut deltas[output_layer_idx];
            let output_error = self.compute_output_error(ouput, target, delta)?;
            current_cost += output_error;

            // Backpropagate through hidden layers
            for layer in (0..self.weights.len()).rev() {
                let gradient_norm = self.compute_layer_gradients(layer, &self.activations, &deltas, weight_gradients, bias_gradients)?;

                max_gradient_norm = max_gradient_norm.max(gradient_norm);

                if layer > 0 {
                    let next_delta = &deltas[layer + 1].clone();
                    self.compute_layer_delta(next_delta, &self.weights[layer], &self.activations[layer], &mut deltas[layer])?;
                }
            }

            // Check for training issues
            if max_gradient_norm > 10.0 {
                return Err(NNError::TrainingError {
                    msg: format!("Gradient norm exploded at batch: {} (norm: {})", batch_idx, max_gradient_norm),
                    cost: Some(current_cost),
                });
            }
        }

        // Apply gradients with momentum and adaptive learning rate
        self.apply_gradients_advanced(weight_gradients, bias_gradients, learning_rate, batch_size, max_gradient_norm)?;

        Ok(())
    }

    fn compute_layer_gradients(
        &self,
        layer: usize,
        layer_outputs: &[Matrix<f32>],
        deltas: &[Matrix<f32>],
        weight_gradients: &mut Vec<Matrix<f32>>,
        bias_gradients: &mut Vec<Matrix<f32>>,
    ) -> Result<f32, NNError> {
        let mut max_gradient: f32 = 0.0;

        // Compute weight gradients
        for i in 0..self.weights[layer].rows {
            for j in 0..self.weights[layer].cols {
                let gradient = layer_outputs[layer][(0, i)] * deltas[layer + 1][(0, j)];
                weight_gradients[layer][(i, j)] += gradient;
                max_gradient = max_gradient.max(gradient.abs());
            }
        }

        // Update bias gradients
        bias_gradients[layer].add(&deltas[layer + 1])?;

        Ok(max_gradient)
    }

    fn compute_layer_delta(
        &self,
        next_delta: &Matrix<f32>,
        weights: &Matrix<f32>,
        layer_output: &Matrix<f32>,
        current_delta: &mut Matrix<f32>,
    ) -> Result<(), NNError> {
        for i in 0..current_delta.cols {
            let mut sum = 0.0;
            for j in 0..next_delta.cols {
                sum += next_delta[(0, j)] * weights[(i, j)];
            }
            current_delta[(0, i)] = sum * self.activation_fn.derivative(layer_output[(0, i)]);
        }
        Ok(())
    }

    fn compute_output_error(&self, output: &Matrix<f32>, target: &Matrix<f32>, delta: &mut Matrix<f32>) -> Result<f32, NNError> {
        let mut error = output.clone();
        error.sub(target)?;

        let mut total_error = 0.0;
        for j in 0..output.cols {
            let err = error[(0, j)];
            total_error += err * err;
            delta[(0, j)] = err * self.activation_fn.derivative(output[(0, j)]);
        }

        Ok(total_error)
    }

    fn apply_gradients_advanced(
        &mut self,
        w_gradients: &mut Vec<Matrix<f32>>,
        b_gradients: &mut Vec<Matrix<f32>>,
        learning_rate: f32,
        batch_size: usize,
        gradient_norm: f32,
    ) -> Result<(), NNError> {
        // Adaptive learning rate based on gradient norm
        let adjusted_rate = if gradient_norm > 1.0 {
            learning_rate / gradient_norm
        } else {
            learning_rate
        };

        let batch_factor = adjusted_rate / batch_size as f32;

        for i in 0..self.weights.len() {
            // Clip and scale gradients
            Self::clip_gradients(&mut w_gradients[i], 1.0);
            Self::clip_gradients(&mut b_gradients[i], 1.0);

            // Apply scaled gradients
            w_gradients[i].apply_fn(|x| x * batch_factor);
            b_gradients[i].apply_fn(|x| x * batch_factor);

            self.weights[i].sub(&w_gradients[i])?;
            self.biases[i].sub(&b_gradients[i])?;
        }
        Ok(())
    }

    pub fn learn(&mut self, epochs: usize, dataset: &DataSet, learning_rate: f32) -> Result<(), NNError> {
        let mut weight_gradients: Vec<Matrix<f32>> = self.weights.iter().map(|w| Matrix::new(w.rows, w.cols)).collect();
        let mut bias_gradients: Vec<Matrix<f32>> = self.biases.iter().map(|b| Matrix::new(b.rows, b.cols)).collect();

        for _ in 0..epochs {
            self.backpropagation(&mut weight_gradients, &mut bias_gradients, dataset, learning_rate)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NeuralNetwork {{\n")?;
        write!(f, "Architecture: {:?}\n", self.arch)?;
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
