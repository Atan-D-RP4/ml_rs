// This is a single 'Neuron' Neural Network that learns to multiply by 2.
// It has 1 input and 1 output.
#![allow(unused)]

use rand::Rng;

const TRAINING_DATA: [[i32; 2]; 5] = [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]];

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

// NOTE: This is different with bias. It will still work
// but the cost will not approach 0.0 but some other value.
fn dcost(w: f32) -> f32 {
    let mut res = 0.0;
    TRAINING_DATA.iter().for_each(|data| {
        let x = data[0] as f32;
        let y = data[1] as f32;
        res += 2.0 * (x * w - y) * x;
    });
    res / TRAINING_DATA.len() as f32
}

pub fn simple_model() {
    let mut w = 0.245;
    let mut b = 0.0;
    println!("Hello, world!");

    println!("w = {:.2}\tb = {:.2}\tcost = {}", w, b, cost(w, b));
    let _eps = 1e-3;
    let rate = 1e-2;
    (0..100).for_each(|_| {
        // let dweight = (cost(w + _eps, b) - cost(w, b)) / _eps;
        // let dbias = (cost(w, b + _eps) - cost(w, b)) / _eps;
        let dweight = dcost(w);
        w -= dweight * rate;
        println!("w = {}\tb = {}\tcost = {}", w, b, cost(w, b));
    });

    (1..5).for_each(|x| {
        println!("{} * 2 = {}", x, (w * x as f32 + b).round());
    });
}
