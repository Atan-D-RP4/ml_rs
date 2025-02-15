use ml_rs::matrix::Matrix;
use ml_rs::nn::{Activation, Architecture, DataSet, NeuralNetwork};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ml_rs::mat_backprop::NeuralNetworkMin::model_run()
    let xor_data = Matrix::from_vec2d(vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ])
    .unwrap();
    let mut dataset = DataSet::new(xor_data, 2)?; // 2 input columns, 1 target column
    let mut nn = NeuralNetwork::new(
        Architecture {
            inputs: dataset.stride,
            layers: &[2, 2],
            outputs: dataset.data.cols - dataset.stride,
        },
        Activation::Tanh,
    );
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

    for _ in 0..2 {
        nn.backpropagation(
            &mut weight_gradients,
            &mut bias_gradients,
            &mut dataset,
            0.1,
        )?
    }

    Ok(())
}
