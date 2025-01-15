fn main() -> Result<(), Box<dyn std::error::Error>> {
    // use ml_rs::complex_gates::Xor;
    // let mut m = Xor::new(&ml_rs::complex_gates::XOR_TRAIN);
    // m.run();

    // use ml_rs::simple_series;
    // simple_series::simple_model();

    {
        // NOTE: Only Matrix implementation
        // The Crude attempt that I still don't follow.
        // Claude had to fix the code to make it work.
        use ml_rs::matrix::Matrix;
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }

        fn forward(
            inputs: &Matrix<f32>,
            weights: &[&Matrix<f32>],
            biases: &[&Matrix<f32>],
        ) -> (Matrix<f32>, Matrix<f32>) {
            // First layer
            let mut hidden = inputs.dot(&weights[0]);
            hidden.add(&biases[0]); // Add bias before activation
            hidden.apply_fn(sigmoid);

            // Output layer
            let mut output = hidden.dot(&weights[1]);
            output.add(&biases[1]); // Add bias before activation
            output.apply_fn(sigmoid);

            (hidden, output)
        }

        fn cost(
            inputs: &Matrix<f32>,
            targets: &Matrix<f32>,
            weights: &[&Matrix<f32>],
            biases: &[&Matrix<f32>],
        ) -> f32 {
            let mut total_cost = 0.0;

            for i in 0..inputs.rows {
                let input_i = inputs.row(i).to_matrix();
                let target_i = targets.row(i).to_matrix();

                let (_, output) = forward(&input_i, weights, biases);
                let mut diff = output;
                diff.sub(&target_i);

                for j in 0..diff.rows {
                    for k in 0..diff.cols {
                        total_cost += diff[(j, k)] * diff[(j, k)];
                    }
                }
            }

            total_cost / (inputs.rows as f32)
        }

        fn gradient(
            inputs: &Matrix<f32>,
            targets: &Matrix<f32>,
            weights: &[&Matrix<f32>],
            biases: &[&Matrix<f32>],
            eps: f32,
        ) -> (Vec<Matrix<f32>>, Vec<Matrix<f32>>) {
            let mut w_grad = vec![
                Matrix::new(weights[0].rows, weights[0].cols),
                Matrix::new(weights[1].rows, weights[1].cols),
            ];
            let mut b_grad = vec![
                Matrix::new(biases[0].rows, biases[0].cols),
                Matrix::new(biases[1].rows, biases[1].cols),
            ];

            let base_cost = cost(inputs, targets, weights, biases);

            // Calculate weight gradients
            for layer in 0..weights.len() {
                for i in 0..weights[layer].rows {
                    for j in 0..weights[layer].cols {
                        let mut temp_w = weights[layer].clone();
                        temp_w[(i, j)] += eps;

                        let temp_weights = if layer == 0 {
                            vec![&temp_w, weights[1]]
                        } else {
                            vec![weights[0], &temp_w]
                        };

                        let perturbed_cost = cost(inputs, targets, &temp_weights, biases);
                        w_grad[layer][(i, j)] = (perturbed_cost - base_cost) / eps;
                    }
                }
            }

            // Calculate bias gradients
            for layer in 0..biases.len() {
                for i in 0..biases[layer].rows {
                    for j in 0..biases[layer].cols {
                        let mut temp_b = biases[layer].clone();
                        temp_b[(i, j)] += eps;

                        let temp_biases = if layer == 0 {
                            vec![&temp_b, biases[1]]
                        } else {
                            vec![biases[0], &temp_b]
                        };

                        let perturbed_cost = cost(inputs, targets, weights, &temp_biases);
                        b_grad[layer][(i, j)] = (perturbed_cost - base_cost) / eps;
                    }
                }
            }

            (w_grad, b_grad)
        }

        fn learn(
            weights: &mut [&mut Matrix<f32>],
            biases: &mut [&mut Matrix<f32>],
            w_grad: &[Matrix<f32>],
            b_grad: &[Matrix<f32>],
            rate: f32,
        ) {
            // Update weights
            for layer in 0..weights.len() {
                for i in 0..weights[layer].rows {
                    for j in 0..weights[layer].cols {
                        weights[layer][(i, j)] -= rate * w_grad[layer][(i, j)];
                    }
                }
            }

            // Update biases
            for layer in 0..biases.len() {
                for i in 0..biases[layer].rows {
                    for j in 0..biases[layer].cols {
                        biases[layer][(i, j)] -= rate * b_grad[layer][(i, j)];
                    }
                }
            }
        }

        // Training data
        let training_inputs = Matrix::from_vec2d(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ]);

        let training_outputs = Matrix::from_vec2d(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]);

        // Initialize weights and biases
        let mut w1 = Matrix::<f32>::new(2, 2); // 2 inputs -> 2 hidden neurons
        let mut b1 = Matrix::<f32>::new(1, 2); // 2 hidden neurons bias
        let mut w2 = Matrix::<f32>::new(2, 1); // 2 hidden neurons -> 1 output
        let mut b2 = Matrix::<f32>::new(1, 1); // 1 output neuron bias

        // Random initialization
        w1.randomize(-1.0, 1.0);
        b1.randomize(-1.0, 1.0);
        w2.randomize(-1.0, 1.0);
        b2.randomize(-1.0, 1.0);

        println!("Initial network state:");
        println!("Weights 1:\n{}", w1);
        println!("Biases 1:\n{}", b1);
        println!("Weights 2:\n{}", w2);
        println!("Biases 2:\n{}", b2);

        // Training parameters
        let rate = 1e-1;
        let eps = 1e-1;

        // Training loop
        for epoch in 0..100_000 {
            let (w_grad, b_grad) = gradient(
                &training_inputs,
                &training_outputs,
                &[&w1, &w2],
                &[&b1, &b2],
                eps,
            );

            learn(
                &mut [&mut w1, &mut w2],
                &mut [&mut b1, &mut b2],
                &w_grad,
                &b_grad,
                rate,
            );

            if epoch % 1000 == 0 {
                let cost_val = cost(
                    &training_inputs,
                    &training_outputs,
                    &[&w1, &w2],
                    &[&b1, &b2],
                );
                println!("Epoch {}: cost = {}", epoch, cost_val);
            }
        }

        // Test the network
        println!("\nFinal results:");
        for i in 0..training_inputs.rows {
            let input = training_inputs.row(i).to_matrix();
            let (_, output) = forward(&input, &[&w1, &w2], &[&b1, &b2]);
            println!(
                "{} XOR {} = {:.3}",
                input[(0, 0)],
                input[(0, 1)],
                output[(0, 0)]
            );
        }
    }

    Ok(())
}

fn xor_model() {
    // NOTE: Framework Implementation
    use ml_rs::nn::{Activation, NeuralNetwork};
    let mut nn = NeuralNetwork::new(&[2, 3, 1], Activation::Tanh);
    nn.randomize(-1.0, 1.0);
    let xor_train = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let init_cost = nn.cost(&xor_train);
    println!("Initial cost: {}", init_cost);

    for epoch in 0..100_000 {
        nn.backpropagation(&xor_train, 1e-1);
        if (epoch + 1) % 1000 == 0 {
            println!("Epoch {}: cost = {}", epoch + 1, nn.cost(&xor_train));
        }
    }

    let final_cost = nn.cost(&xor_train);
    println!("Final cost: {}", final_cost);

    // Test against inputs
    for i in 0..2 {
        for j in 0..2 {
            let input = vec![i as f32, j as f32];
            let output = nn.predict(&input)[(0, 0)];
            println!("{:?} => {}", input, output);
        }
    }
}
