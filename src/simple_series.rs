// This is a single 'Neuron' Neural Network that learns to multiply by 2.
use rand::Rng;

const TRAINING_DATA: [[i32; 2]; 4] = [[0, 0], [1, 2], [2, 4], [3, 6]];

fn cost(w: f32, bias: f32) -> f32 {
    let mut cost = 0.0;
    TRAINING_DATA.iter().for_each(|data| {
        let x = data[0];
        let y = x as f32 * w + bias;
        let d = y - data[1] as f32;
        cost += d * d;
    });
    cost / TRAINING_DATA.len() as f32
}

pub fn simple_model() {
    let mut w = rand::thread_rng().gen_range(-100.0..100.0);
    let mut b = 0.0;
    println!("Hello, world!");

    println!("{} {} {}", w, b, cost(w, b));
    let eps = 1e-3;
    let rate = 1e-1;
    (0..1000).for_each(|_| {
        let dweight = (cost(w + eps, b) - cost(w, b)) / eps;
        let dbias = (cost(w, b + eps) - cost(w, b)) / eps;
        w -= dweight * rate;
        b -= dbias * rate;
        println!("{} {} {}", w, b, cost(w, b));
    });

    (1..101).for_each(|x| {
        println!("{} * 2 = {}", x, (w * x as f32 + b).round());
    });
}
