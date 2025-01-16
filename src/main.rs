use ml_rs::matrix::Matrix;
use rand::Rng;

#[derive(Debug)]
struct Network2 {
    w1: Matrix<f32>,
    b1: Matrix<f32>,
    w2: Matrix<f32>,
    b2: Matrix<f32>,
    hidden_cache: Matrix<f32>,
    output_cache: Matrix<f32>,
}

impl Network2 {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut w1 = Matrix::new(input_size, hidden_size);
        let mut w2 = Matrix::new(hidden_size, output_size);
        let mut b1 = Matrix::new(1, hidden_size);
        let mut b2 = Matrix::new(1, output_size);

        // Initialize with small random weights
        let w1_bound = (2.0 / (input_size as f32)).sqrt();
        let w2_bound = (2.0 / (hidden_size as f32)).sqrt();
        w1.apply_fn(|_| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * w1_bound);
        w2.apply_fn(|_| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * w2_bound);
        b1.fill(0.0);
        b2.fill(0.0);

        Network2 {
            w1,
            b1,
            w2,
            b2,
            hidden_cache: Matrix::new(1, hidden_size),
            output_cache: Matrix::new(1, output_size),
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn forward(&mut self, input: &Matrix<f32>) -> &Matrix<f32> {
        // First layer
        let mut hidden = input.dot(&self.w1);
        hidden.add(&self.b1);
        hidden.apply_fn(Self::sigmoid);
        self.hidden_cache = hidden;

        // Output layer
        let mut output = self.hidden_cache.dot(&self.w2);
        output.add(&self.b2);
        output.apply_fn(Self::sigmoid);
        self.output_cache = output;

        &self.output_cache
    }

    fn cost(&mut self, inputs: &Matrix<f32>, targets: &Matrix<f32>) -> f32 {
        let output = self.forward(inputs);
        let mut cost = 0.0;

        for i in 0..targets.rows {
            for j in 0..targets.cols {
                let diff = output[(i, j)] - targets[(i, j)];
                cost += diff * diff;
            }
        }

        cost / (targets.rows as f32)
    }

    fn finite_diff(&mut self, inputs: &Matrix<f32>, targets: &Matrix<f32>, eps: f32) -> Network2 {
        let base_cost = self.cost(inputs, targets);
        let mut gradients = Network2::new(self.w1.rows, self.w1.cols, self.w2.cols);

        // Compute gradients for w1
        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1[(i, j)] += eps;
                let new_cost = self.cost(inputs, targets);
                gradients.w1[(i, j)] = (new_cost - base_cost) / eps;
                self.w1[(i, j)] -= eps;
            }
        }

        // Compute gradients for w2
        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                self.w2[(i, j)] += eps;
                let new_cost = self.cost(inputs, targets);
                gradients.w2[(i, j)] = (new_cost - base_cost) / eps;
                self.w2[(i, j)] -= eps;
            }
        }

        // Compute gradients for b1
        for i in 0..self.b1.cols {
            self.b1[(0, i)] += eps;
            let new_cost = self.cost(inputs, targets);
            gradients.b1[(0, i)] = (new_cost - base_cost) / eps;
            self.b1[(0, i)] -= eps;
        }

        // Compute gradients for b2
        for i in 0..self.b2.cols {
            self.b2[(0, i)] += eps;
            let new_cost = self.cost(inputs, targets);
            gradients.b2[(0, i)] = (new_cost - base_cost) / eps;
            self.b2[(0, i)] -= eps;
        }

        gradients
    }

    fn learn(&mut self, gradients: &Network2, learning_rate: f32) {
        // Update weights and biases using computed gradients
        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1[(i, j)] -= learning_rate * gradients.w1[(i, j)];
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                self.w2[(i, j)] -= learning_rate * gradients.w2[(i, j)];
            }
        }

        for i in 0..self.b1.cols {
            self.b1[(0, i)] -= learning_rate * gradients.b1[(0, i)];
        }

        for i in 0..self.b2.cols {
            self.b2[(0, i)] -= learning_rate * gradients.b2[(0, i)];
        }
    }

    fn train_epoch(
        &mut self,
        inputs: &Matrix<f32>,
        targets: &Matrix<f32>,
        eps: f32,
        learning_rate: f32,
    ) -> f32 {
        let gradients = self.finite_diff(inputs, targets, eps);
        self.learn(&gradients, learning_rate);
        self.cost(inputs, targets)
    }
}

fn model_run() {
    use ml_rs::nn::DataSet;
    let ds = DataSet::new(
        Matrix::from_vec2d(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]),
        2,
    );
    println!("Data: {}", ds.data);

    let inputs = ds.inputs_as_matrix();
    println!("Inputs: {}", inputs);

    let targets = ds.targets_as_matrix();
    println!("Targets: {}", targets);

    let mut network = Network2::new(2, 3, 1);
    let eps = 1e-1;
    let learning_rate = 1e-1;
    let epochs = 100_000;

    println!("Initial cost: {}", network.cost(&inputs, &targets));

    for epoch in 0..epochs {
        let cost = network.train_epoch(&inputs, &targets, eps, learning_rate);

        if epoch % 1000 == 0 {
            println!("Epoch {}: cost = {}", epoch, cost);
        }
    }

    // Test the network
    for i in 0..2 {
        for j in 0..2 {
            let input = Matrix::from_vec2d(vec![vec![i as f32, j as f32]]);
            let output = network.forward(&input);
            println!("{} XOR {} = {}", i, j, output[(0, 0)]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    model_run();
    Ok(())
}
