use rand::Rng;

type Sample = [f32; 3];

#[derive(Copy, Clone, Debug)]
struct Xor {
    or_w1: f32,
    or_w2: f32,
    or_b: f32,
    nand_w1: f32,
    nand_w2: f32,
    nand_b: f32,
    and_w1: f32,
    and_w2: f32,
    and_b: f32,

    sample: [Sample; 4],
}

const XOR_TRAIN: [Sample; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];

const OR_TRAIN: [Sample; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

const AND_TRAIN: [Sample; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
];

const NAND_TRAIN: [Sample; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];

const NOR_TRAIN: [Sample; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
];

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Xor {
    fn new(sample: &[Sample; 4]) -> Self {
        let mut rng = rand::thread_rng();
        Xor {
            or_w1: rng.gen(),
            or_w2: rng.gen(),
            or_b: rng.gen(),
            nand_w1: rng.gen(),
            nand_w2: rng.gen(),
            nand_b: rng.gen(),
            and_w1: rng.gen(),
            and_w2: rng.gen(),
            and_b: rng.gen(),

            sample: *sample,
        }
    }

    fn forward(&self, x1: f32, x2: f32) -> f32 {
        let a = sigmoid(self.or_w1 * x1 + self.or_w2 * x2 + self.or_b);
        let b = sigmoid(self.nand_w1 * x1 + self.nand_w2 * x2 + self.nand_b);
        sigmoid(a * self.and_w1 + b * self.and_w2 + self.and_b)
    }

    fn cost(&self) -> f32 {
        let train = &self.sample;
        let result: f32 = train
            .iter()
            .map(|&[x1, x2, expected]| {
                let y = self.forward(x1, x2);
                let d = y - expected;
                d * d
            })
            .sum();
        result / train.len() as f32
    }

    fn learn(&mut self, g: &Xor, rate: f32) {
            self.or_w1 = self.or_w1 - rate * g.or_w1;
            self.or_w2 = self.or_w2 - rate * g.or_w2;
            self.or_b = self.or_b - rate * g.or_b;
            self.nand_w1 = self.nand_w1 - rate * g.nand_w1;
            self.nand_w2 = self.nand_w2 - rate * g.nand_w2;
            self.nand_b = self.nand_b - rate * g.nand_b;
            self.and_w1 = self.and_w1 - rate * g.and_w1;
            self.and_w2 = self.and_w2 - rate * g.and_w2;
            self.and_b = self.and_b - rate * g.and_b;
            self.sample = self.sample;
    }

    fn finite_diff(&self, eps: f32) -> Self {
        let c = self.cost();
        let mut g = *self;

        // Compute gradients for each parameter
        macro_rules! compute_gradient {
            ($field:ident) => {{
                let mut m = *self;
                m.$field += eps;
                g.$field = (m.cost() - c) / eps;
            }};
        }

        compute_gradient!(or_w1);
        compute_gradient!(or_w2);
        compute_gradient!(or_b);
        compute_gradient!(nand_w1);
        compute_gradient!(nand_w2);
        compute_gradient!(nand_b);
        compute_gradient!(and_w1);
        compute_gradient!(and_w2);
        compute_gradient!(and_b);

        g
    }

    fn run(&mut self) {
        let eps = 1e-1;
        let rate = 1e-1;

        // Training
        for _ in 0..100_000 {
            let g = self.finite_diff(eps);
            self.learn(&g, rate);
        }

        println!("cost = {}", self.cost());
        println!("------------------------------");

        // Test operation of Corresponding data set
        for i in 0..2 {
            for j in 0..2 {
                println!("{} ^ {} = {}", i, j, self.forward(i as f32, j as f32)/*.round()*/);
            }
        }

        println!("------------------------------");
        println!("\"OR\" neuron:");
        for i in 0..2 {
            for j in 0..2 {
                println!(
                    "{} | {} = {}",
                    i,
                    j,
                    sigmoid(self.or_w1 * i as f32 + self.or_w2 * j as f32 + self.or_b)
                );
            }
        }

        println!("------------------------------");
        println!("\"NAND\" neuron:");
        for i in 0..2 {
            for j in 0..2 {
                println!(
                    "~({} & {}) = {}",
                    i,
                    j,
                    sigmoid(self.nand_w1 * i as f32 + self.nand_w2 * j as f32 + self.nand_b)
                );
            }
        }

        println!("------------------------------");
        println!("\"AND\" neuron:");
        for i in 0..2 {
            for j in 0..2 {
                println!(
                    "{} & {} = {}",
                    i,
                    j,
                    sigmoid(self.and_w1 * i as f32 + self.and_w2 * j as f32 + self.and_b)
                );
            }
        }
    }
}
