use crate::matrix::{Matrix, MatrixError};
use crate::nn::{Activation, Architecture, DataSet};
use rand::distributions::Distribution;

pub struct NeuralNetworkMin {
    pub architecture: Vec<usize>,
    pub weights: Vec<Matrix<f32>>,
    pub biases: Vec<Matrix<f32>>,
    activation_fn: Activation,
    activations: Vec<Matrix<f32>>,
}

impl NeuralNetworkMin {
    pub fn new(arch: Architecture, activation: Activation) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut architecture = Vec::new();
        architecture.push(arch.inputs);
        arch.layers.iter().for_each(|neuron_count| {
            architecture.push(*neuron_count);
        });
        architecture.push(arch.outputs);

        for i in 0..architecture.len() - 1 {
            weights.push(Matrix::new(architecture[i], architecture[i + 1]));
            biases.push(Matrix::new(1, architecture[i + 1]));
        }

        let mut activations = Vec::with_capacity(architecture.len());
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
        self.forward(input)?;
        Ok(self.activations.last().unwrap().clone())
    }

    pub fn backpropagation(
        &mut self,
        dataset: &DataSet,
        learning_rate: f32,
    ) -> Result<(), MatrixError> {
        let inputs = dataset.inputs();
        let targets = dataset.targets();
        let batch_size = inputs.len();

        let base_cost = self.cost(&dataset)?;
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

        for i in 0..inputs.len() {
            // Iterate on all samples
            let input = &inputs[i];
            let target = &targets[i];
            self.forward(input)?;
            let activations = &self.activations;

            // Calculate layer errors
            let mut errors = vec![Matrix::<f32>::new(1, *self.architecture.last().unwrap())];
            let output_error = activations.last().unwrap();

            for j in 0..output_error.cols {
                let diff = output_error[(0, j)] - target[(0, j)];
                errors[0][(0, j)] = diff * self.activation_fn.derivative(output_error[(0, j)]);
            }

            // Backpropagating errors by unrolling chain rule
            for layer in (0..(self.weights.len())).rev() {
                let curr_error = &errors[0];

                // Calculate gradients
                for i in 0..self.weights[layer].rows {
                    for j in 0..self.weights[layer].cols {
                        weight_gradients[layer][(i, j)] +=
                            curr_error[(0, j)] * activations[layer][(0, i)];
                    }
                }

                for i in 0..self.biases[layer].rows {
                    for j in 0..self.biases[layer].cols {
                        bias_gradients[layer][(i, j)] += curr_error[(0, j)];
                    }
                }

                if layer > 0 {
                    let mut new_error = Matrix::new(1, self.architecture[layer]);
                    for i in 0..new_error.cols {
                        let mut sum = 0.0;
                        for j in 0..curr_error.cols {
                            sum += self.weights[layer][(i, j)] * curr_error[(0, j)];
                        }
                        new_error[(0, i)] =
                            sum * self.activation_fn.derivative(activations[layer][(0, i)]);
                    }
                    errors[0] = new_error;
                }
            }
        }

        self.learn(weight_gradients, bias_gradients, learning_rate, batch_size);
        Ok(())
    }

    fn forward(&mut self, input: &Matrix<f32>) -> Result<(), MatrixError> {
        // Set input layer
        let activations = &mut self.activations;
        activations[0] = input.clone();
        activations.iter_mut().skip(1).for_each(|a| a.fill(0.0));

        // Forward propagation
        for i in 0..self.weights.len() {
            let activation = activations[i].dot(&self.weights[i])?;
            activations[i + 1] = activation;
            activations[i + 1].add(&self.biases[i])?;
            activations[i + 1].apply_activation(self.activation_fn);
        }
        Ok(())
    }

    pub fn cost(&mut self, dataset: &DataSet) -> Result<f32, MatrixError> {
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

    fn compute_numerical_gradient(
        &mut self,
        dataset: &DataSet,
        epsilon: f32,
    ) -> Result<(Vec<Matrix<f32>>, Vec<Matrix<f32>>), MatrixError> {
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

        // Compute gradients for weights
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].elements.len() {
                // Central difference approximation
                let original = self.weights[i].elements[j];

                // Positive perturbation
                self.weights[i].elements[j] = original + epsilon;
                let pos_cost = self.cost(dataset)?;

                // Negative perturbation
                self.weights[i].elements[j] = original - epsilon;
                let neg_cost = self.cost(dataset)?;

                // Restore original value
                self.weights[i].elements[j] = original;

                // Symmetric difference gradient
                weight_gradients[i].elements[j] = (pos_cost - neg_cost) / (2.0 * epsilon);
            }
        }

        // Compute gradients for biases (similar to weights)
        for i in 0..self.biases.len() {
            for j in 0..self.biases[i].elements.len() {
                let original = self.biases[i].elements[j];

                self.biases[i].elements[j] = original + epsilon;
                let pos_cost = self.cost(dataset)?;

                self.biases[i].elements[j] = original - epsilon;
                let neg_cost = self.cost(dataset)?;

                self.biases[i].elements[j] = original;

                bias_gradients[i].elements[j] = (pos_cost - neg_cost) / (2.0 * epsilon);
            }
        }

        Ok((weight_gradients, bias_gradients))
    }

    pub fn finite_diff(
        &mut self,
        dataset: &DataSet,
        learning_rate: f32,
    ) -> Result<(), MatrixError> {
        const EPSILON: f32 = 1e-1; // Small, stable perturbation
                                   // const REGULARIZATION: f32 = 1e-3; // Optional L2 regularization

        let base_cost = self.cost(dataset)?;
        let (weight_gradients, bias_gradients) =
            self.compute_numerical_gradient(dataset, EPSILON)?;

        // Optional: Gradient clipping to prevent exploding gradients
        // let max_gradient = 1.0;
        // let weight_gradients = weight_gradients.into_iter().map(|mut grad| {
        //     grad.elements.iter_mut().for_each(|g| {
        //         *g = g.clamp(-max_gradient, max_gradient);
        //     });
        //     grad
        // });
        // let weight_gradients = weight_gradients.collect::<Vec<_>>();

        // Batch size based on dataset inputs
        let batch_size = dataset.inputs().len();

        self.learn(weight_gradients, bias_gradients, learning_rate, batch_size);

        let final_cost = self.cost(dataset)?;

        // Optional: Cost validation
        if final_cost > base_cost * 1.1 {
            println!(
                "Warning: Cost increased. Base: {}, Final: {}",
                base_cost, final_cost
            );
        }

        Ok(())
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

    pub fn model_run() -> Result<(), Box<dyn std::error::Error>> {
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
        let ds = DataSet::new(adder_data, 3)?;

        let arch = Architecture {
            inputs: ds.stride,
            layers: &[6],
            outputs: ds.data.cols - ds.stride,
        }; // NOTE: [2, 3, 1] - 1 hidden layers
        let mut nn = NeuralNetworkMin::new(arch, Activation::Relu(1e-1));
        println!("Initialising parameters...");
        nn.init_parameters(ds.stride)?;
        println!("{}", nn);

        let initial_cost = nn.cost(&ds)?;
        println!("Initial Cost: {}", initial_cost);

        let start = std::time::Instant::now();
        println!("Timer Start");

        (0..40_000).for_each(|i| {
            // nn.finite_diff(&ds, 1e-1).expect("Error in finite_diff");
            nn.backpropagation(&ds, 1e-1)
                .expect("Error while backpropagating");
            if i % 1000 == 0 {
                println!("{} :: Cost: {}", i, nn.cost(&ds).expect("Error in cost_fn"));
            }
        });
        let final_cost = nn.cost(&ds)?;
        println!("Final Cost: {}", final_cost);
        println!("Time Elapsed: {:?}", start.elapsed());

        Ok(())
    }
}

impl std::fmt::Display for NeuralNetworkMin {
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
