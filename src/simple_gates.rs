// This is a simple implementation of AND, OR, NAND gates using a simple neural network
// consisting of a single 'Neuron'

const TRAINING_SET_OR: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];

const TRAINING_SET_AND: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
];

const TRAINING_SET_NAND: [[f32; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
];

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// X ^ Y = (X|Y) & !(X & Y) -> This is XOR expressed in terms of OR and AND

fn cost_fn(weights: &[f32], bias: f32, data_set: &[[f32; 3]]) -> f32 {
    let mut cost = 0.0;
    data_set.iter().for_each(|data| {
        let x1 = data[0];
        let x2 = data[1];
        let y = data[2];
        let z = sigmoid(weights[0] * x1 + weights[1] * x2 + bias);
        cost += (y - z) * (y - z);
    });
    cost / data_set.len() as f32
}

fn train_model(data_set: &[[f32; 3]]) -> (Vec<f32>, f32) {
    let mut ws = vec![rand::random::<f32>(); 2];
    let mut bias = rand::random::<f32>();
    let eps = 1e-2;
    let rate = 1e-1;

    let range = 0..1000 * 1000;
    range.for_each(|_| {
        let cost = cost_fn(&ws, bias, data_set);
        // println!("weights: {:?}, bias: {}, cost: {}", ws, bias, cost);
        let dw1 = cost_fn(&[ws[0] + eps, ws[1]], bias, data_set) - cost;
        let dw2 = cost_fn(&[ws[0], ws[1] + eps], bias, data_set) - cost;
        let dbias = cost_fn(&ws, bias + eps, data_set) - cost;

        ws[0] -= dw1 * rate;
        ws[1] -= dw2 * rate;
        bias -= dbias * rate;
    });
    println!(
        "weights: {:?}, bias: {}, cost: {}",
        ws,
        bias,
        cost_fn(&ws, bias, data_set)
    );
    (0..2).for_each(|i| {
        (0..2).for_each(|j| {
            println!(
                "{} | {} = {}",
                i,
                j,
                sigmoid(ws[0] * i as f32 + ws[1] * j as f32 + bias)
            );
        });
    });
    (ws, bias)
}

pub fn gates() -> Result<(), Box<dyn std::error::Error>> {
    let _model_and = train_model(&TRAINING_SET_AND);
    let _model_or = train_model(&TRAINING_SET_OR);
    let _model_nand = train_model(&TRAINING_SET_NAND);
    Ok(())
}
