use ml_rs::matrix;
use ml_rs::matrix::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // use ml_rs::complex_gates::Xor;
    // let mut m = Xor::new(&XOR_TRAIN);
    // m.run()

    // use ml_rs::simple_series;
    // simple_series::simple_model();

    let matrix_a = Matrix::<f32>::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
    let matrix_b = Matrix::<f32>::from_vec(vec![3.0, 4.0, 3.0, 4.0], 2, 2).unwrap();
    let matrix_c = (matrix_a * matrix_b).unwrap();
    println!("{}", matrix_c);

    let matrix_a = matrix![[1, 2], [2, 2]];
    let or_data = matrix![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let matrix_b = Matrix::from_vec2(vec![vec![1, 2], vec![3, 4]]).unwrap();
    println!("{}", (matrix_a * matrix_b).unwrap());
    println!("{}", or_data);

    let (x, y) = or_data.split_vert(4).unwrap_or_else(|e| {
        println!("{:?}", e);
        (Matrix::new(0, 0), Matrix::new(0, 0))
    });
    println!("{}", x);
    println!("{}", y);

    let (x, y) = or_data.split_horz(4).unwrap_or_else(|e| {
        println!("{:?}", e);
        (Matrix::new(0, 0), Matrix::new(0, 0))
    });
    println!("{}", x);
    println!("{}", y);
    Ok(())
}
