use ml_rs::matrix;
use ml_rs::matrix::Matrix;
use rand::{random, Rng};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_mat(m: &Matrix<f32>) -> Matrix<f32> {
    let x = m.data.clone().iter().map(|x| sigmoid(*x)).collect();
    let (rows, cols) = m.dimensions();
    Matrix::from_vec(x, rows, cols).unwrap()
}

fn forward(model: &NeuralNetwork, inputs: &Matrix<f32>) -> Matrix<f32> {
    let mut out = inputs.clone();
    let a = {
        out = match (model.weights[1].clone() * out.clone()) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("a * b:\n{}\n{}", &out, model.weights[0]);
                panic!("Error: {:?}", e);
            }
        };
        out = (out + Matrix::from_vec(vec![model.bias[0]], 1, 1).unwrap()).unwrap();
        out = sigmoid_mat(&out);
        out
    };
    let mut out = inputs.clone();
    let b = {
        out = (out * model.weights[1].clone()).unwrap();
        out = (out + Matrix::from_vec(vec![model.bias[1]], 1, 1).unwrap()).unwrap();
        out = sigmoid_mat(&out);
        out
    };

    let mut out = Matrix::from_vec(vec![a.data[0], b.data[0]], 1, 2).unwrap();
    let out = {
        out = (out * model.weights[2].clone()).unwrap();
        out = (out + Matrix::from_vec(vec![model.bias[2]], 1, 1).unwrap()).unwrap();
        out = sigmoid_mat(&out);
        out
    };
    out
}

fn cost(model: &NeuralNetwork, training_data: &Matrix<f32>) -> f32 {
    let mut cost = 0.0;
    training_data.iter_rows().for_each(|row| {
        let row = Matrix::from_vec(row.to_vec(), 1, 3).unwrap();
        let (inputs, expected) = row.split_vert(2).unwrap();
        let out = forward(model, &inputs);
        cost += (out - expected).unwrap().data[0].powi(2);
    });
    cost / training_data.dimensions().0 as f32
}

fn finite_diff(model: &NeuralNetwork, eps: f32) -> NeuralNetwork {
    let mut new_model = model.clone();
    for i in 0..model.weights.len() {
        let weight_set = model.weights[i].clone();
        let new_weight_set = (weight_set
            + Matrix::from_vec(vec![eps, eps], 1, 2).expect("eps to 1x1 mat failed"))
        .expect("addition failed");
        new_model.weights[i] = new_weight_set;
    }
    for i in 0..model.bias.len() {
        let bias = model.bias[i];
        new_model.bias[i] = bias + eps;
    }
    new_model
}

fn learn(model: &mut NeuralNetwork, diff: NeuralNetwork, rate: f32) {
    for i in 0..model.weights.len() {
        let weight_set = model.weights[i].clone();
        let diff_weight_set = diff.weights[i].clone();
        model.weights[i] = (weight_set
            - (diff_weight_set
                * Matrix::from_vec(vec![rate; 4], 2, 2).expect("rate to vec failed"))
            .expect("* with rate mat failed"))
        .unwrap();
    }
    for i in 0..model.bias.len() {
        let bias = model.bias[i];
        let diff_bias = diff.bias[i];
        model.bias[i] = bias - diff_bias * rate;
    }
}

#[derive(Clone, Debug)]
struct NeuralNetwork {
    weights: Vec<Matrix<f32>>,
    bias: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // use ml_rs::complex_gates::Xor;
    // let mut m = Xor::new(&XOR_TRAIN);
    // m.run()

    // use ml_rs::simple_series;
    // simple_series::simple_model();

    let xor_data = matrix![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ];
    let training_data = xor_data;
    let mut model = NeuralNetwork {
        weights: vec![
            matrix!(gen 1, 2, || rand::thread_rng().gen_range(0.0..10.0)),
            matrix!(gen 1, 2, || rand::thread_rng().gen_range(0.0..10.0)),
            matrix!(gen 1, 2, || rand::thread_rng().gen_range(0.0..10.0)),
        ],
        bias: vec![rand::random::<f32>(); 2],
    };
    let eps = 1e-1;
    let rate = 1e-1;

    (0..100).for_each(|_| {
        let g = finite_diff(&model, eps);
        learn(&mut model, g, rate);
    });

    let (inputs, _) = training_data.split_vert(2).unwrap();
    println!("Cost: {}", cost(&model, &training_data));

    // Test operation of Corresponding data set
    for i in 0..2 {
        for j in 0..2 {
            println!("{} ^ {} = {}", i, j, forward(&model, &inputs),);
        }
    }

    Ok(())
}
