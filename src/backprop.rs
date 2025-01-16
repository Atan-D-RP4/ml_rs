use crate::matrix::Matrix;
use rand::Rng;

#[derive(Debug)]
struct Network {
    w1: Matrix<f32>,
    b1: Matrix<f32>,
    w2: Matrix<f32>,
    b2: Matrix<f32>,
    hidden_cache: Matrix<f32>,
    output_cache: Matrix<f32>,
    // Add caches for pre-activation values
    z1_cache: Matrix<f32>, // Pre-activation cache for hidden layer
    z2_cache: Matrix<f32>, // Pre-activation cache for output layer
    grad_w1: Matrix<f32>,
    grad_b1: Matrix<f32>,
    grad_w2: Matrix<f32>,
    grad_b2: Matrix<f32>,
}

impl Network {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        // Use He initialization for ReLU (if switching to ReLU)
        // Or keep Xavier/Glorot for sigmoid but with proper scaling
        let w1_bound = (2.0 / (input_size as f32)).sqrt();
        let w2_bound = (2.0 / (hidden_size as f32)).sqrt();

        let mut w1 = Matrix::new(input_size, hidden_size);
        let mut w2 = Matrix::new(hidden_size, output_size);
        let mut b1 = Matrix::new(1, hidden_size);
        let mut b2 = Matrix::new(1, output_size);

        // Initialize with smaller weights to prevent saturation
        w1.apply_fn(|_| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * w1_bound);
        w2.apply_fn(|_| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * w2_bound);
        // Initialize biases to zero instead of random values
        b1.fill(0.0);
        b2.fill(0.0);

        Network {
            w1,
            b1,
            w2,
            b2,
            hidden_cache: Matrix::new(1, hidden_size),
            output_cache: Matrix::new(1, output_size),
            z1_cache: Matrix::new(1, hidden_size),
            z2_cache: Matrix::new(1, output_size),
            grad_w1: Matrix::new(input_size, hidden_size),
            grad_b1: Matrix::new(1, hidden_size),
            grad_w2: Matrix::new(hidden_size, output_size),
            grad_b2: Matrix::new(1, output_size),
        }
    }

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        // Clip values to prevent overflow
        let x = if x > 15.0 {
            15.0
        } else if x < -15.0 {
            -15.0
        } else {
            x
        };
        1.0 / (1.0 + (-x).exp())
    }

    #[inline(always)]
    fn sigmoid_derivative(x: f32) -> f32 {
        let s = Self::sigmoid(x);
        s * (1.0 - s)
    }

    fn forward(&mut self, input: &Matrix<f32>) -> &Matrix<f32> {
        assert_eq!(input.cols, self.w1.rows, "Input dimension mismatch");

        // First layer
        let mut z1 = input.dot(&self.w1);
        z1.add(&self.b1);
        self.z1_cache = z1.clone(); // Store pre-activation

        let mut hidden = z1;
        hidden.apply_fn(Self::sigmoid);
        self.hidden_cache = hidden;

        // Output layer
        let mut z2 = self.hidden_cache.dot(&self.w2);
        z2.add(&self.b2);
        self.z2_cache = z2.clone(); // Store pre-activation

        let mut output = z2;
        output.apply_fn(Self::sigmoid);
        self.output_cache = output;

        &self.output_cache
    }

    fn backprop(&mut self, input: &Matrix<f32>, target: &Matrix<f32>, batch_size: f32) {
        self.grad_w1.fill(0.0);
        self.grad_b1.fill(0.0);
        self.grad_w2.fill(0.0);
        self.grad_b2.fill(0.0);

        let output = self.forward(input);

        // Output layer gradients using stored pre-activation values
        let mut output_delta = output.clone();
        output_delta.sub(target);
        output_delta.apply_with_matrix(&self.z2_cache, |d, z| {
            d * Self::sigmoid_derivative(z) / batch_size
        });

        // Hidden layer gradients
        let mut w2_transpose = self.w2.clone();
        w2_transpose.transpose();
        let mut hidden_delta = output_delta.dot(&w2_transpose);
        hidden_delta.apply_with_matrix(&self.z1_cache, |d, z| d * Self::sigmoid_derivative(z));

        // Compute gradients
        let mut input_t = input.clone();
        input_t.transpose();
        let mut hidden_cache_transpose = self.hidden_cache.clone();
        hidden_cache_transpose.transpose();

        self.grad_w2 = hidden_cache_transpose.dot(&output_delta);
        self.grad_w1 = input_t.dot(&hidden_delta);

        // Compute bias gradients by summing across all examples
        for i in 0..output_delta.rows {
            for j in 0..output_delta.cols {
                self.grad_b2[(0, j)] += output_delta[(i, j)];
            }
        }
        for i in 0..hidden_delta.rows {
            for j in 0..hidden_delta.cols {
                self.grad_b1[(0, j)] += hidden_delta[(i, j)];
            }
        }
    }

    fn update(&mut self, learning_rate: f32) {
        // Add gradient clipping
        let clip_threshold = 1.0;

        let clip = |x: f32| {
            if x > clip_threshold {
                clip_threshold
            } else if x < -clip_threshold {
                -clip_threshold
            } else {
                x
            }
        };

        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1[(i, j)] -= learning_rate * clip(self.grad_w1[(i, j)]);
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                self.w2[(i, j)] -= learning_rate * clip(self.grad_w2[(i, j)]);
            }
        }

        for i in 0..self.b1.cols {
            self.b1[(0, i)] -= learning_rate * clip(self.grad_b1[(0, i)]);
        }

        for i in 0..self.b2.cols {
            self.b2[(0, i)] -= learning_rate * clip(self.grad_b2[(0, i)]);
        }
    }

    fn train_epoch(
        &mut self,
        inputs: &Matrix<f32>,
        targets: &Matrix<f32>,
        learning_rate: f32,
    ) -> f32 {
        let batch_size = inputs.rows as f32;
        self.backprop(inputs, targets, batch_size);
        self.update(learning_rate);

        // Calculate cross-entropy loss instead of MSE
        let output = self.forward(inputs);
        let mut cost = 0.0;
        for i in 0..targets.rows {
            for j in 0..targets.cols {
                let y = targets[(i, j)];
                let y_pred = output[(i, j)].max(1e-15).min(1.0 - 1e-15);
                cost += -(y * y_pred.ln() + (1.0 - y) * (1.0 - y_pred).ln());
            }
        }
        cost / batch_size
    }
}

fn network1() {
    let inputs = Matrix::from_vec2d(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);

    let targets = Matrix::from_vec2d(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]);

    let mut network = Network::new(2, 4, 1); // Increased hidden layer size

    // Reduced learning rate and increased epochs
    let learning_rate = 1e-1 * 5.0;
    let epochs = 100_000;
    println!(
        "Initial Cost: {}",
        network.train_epoch(&inputs, &targets, 0.05)
    );

    for epoch in 0..epochs {
        let cost = network.train_epoch(&inputs, &targets, learning_rate);

        if epoch % 1000 == 0 {
            println!("Epoch {}: cost = {}", epoch, cost);
        }
    }

    for i in 0..2 {
        for j in 0..2 {
            let input = Matrix::from_vec2d(vec![vec![i as f32, j as f32]]);
            let output = network.forward(&input);
            println!("{} XOR {} = {}", i, j, output[(0, 0)]);
        }
    }
}
