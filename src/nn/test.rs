use super::*;
use crate::{matrix::Matrix, nn::Activation};

#[test]
fn test_bin_op_nn() -> Result<(), Box<dyn std::error::Error>> {
    // Create XOR training data using DataSet
    let xor_data = Matrix::from_vec2d(vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]]).unwrap();
    let dataset = DataSet::new(xor_data, 2)?; // 2 input columns, 1 target column

    // NOTE: arch -> [inputs, [neurons in each layer]..., outputs]
    // NOTE: arch can be [2, 2, 1, 2] or [2, 5, 5, 7, 2]
    // NOTE: [2, 3, 1] - 1 hidden layers
    let arch = Architecture::with(dataset.stride, &[2, 2], dataset.data.cols - dataset.stride);
    let mut nn = NeuralNetwork::new(arch, Activation::Sigmoid)?;
    nn.init_parameters(dataset.stride)?;

    let initial_cost = nn.cost(&dataset)?;
    println!("Initial Cost: {}", initial_cost);

    let start = std::time::Instant::now();
    // Train for a few epochs
    nn.learn(50000, &dataset, 1e0)?;
    println!("Time Elapsed: {:?}", start.elapsed());

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

    // Test predictions
    let (inputs, targets) = (dataset.inputs_as_matrix().to_vec2d(), dataset.targets_as_matrix().to_vec2d());

    let arch = Architecture::with(dataset.stride, &[6], dataset.data.cols - dataset.stride);
    let mut nn = NeuralNetwork::new(arch, Activation::Sigmoid)?;
    println!("Initialising parameters...");
    nn.init_parameters(dataset.stride)?;

    let initial_cost = nn.cost(&dataset)?;
    println!("Initial Cost: {}", initial_cost);

    let start = std::time::Instant::now();
    nn.learn(50000, &dataset, 1e0)?;
    let final_cost = nn.cost(&dataset)?;
    println!("Time Elapsed: {:?}", start.elapsed());

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
