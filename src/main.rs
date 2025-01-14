use ml_rs::nn::{Activation, NeuralNetwork};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // use ml_rs::complex_gates::Xor;
    // let mut m = Xor::new(&XOR_TRAIN);
    // m.run()

    // use ml_rs::simple_series;
    // simple_series::simple_model();

    {
        let mut nn = NeuralNetwork::new(&[2, 3, 1], Activation::Sigmoid);
        let xor_train = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let init_cost = nn.compute_loss(&xor_train);
        println!("Initial cost: {}", init_cost);

        for epoch in 0..100000 {
            nn.train(&xor_train, 0.1, 4);
            if (epoch + 1) % 1000 == 0 {
                println!("Epoch {}: cost = {}", epoch + 1, nn.compute_loss(&xor_train));
            }
        }

        let final_cost = nn.compute_loss(&xor_train);
        println!("Final cost: {}", final_cost);

        // Test against inputs
        for i in 0..2 {
            for j in 0..2 {
                let input = vec![i as f32, j as f32];
                let output = nn.predict(&input);
                println!("{:?} => {:?}", input, output);
            }
        }
    }

    Ok(())
}
