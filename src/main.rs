use ml_rs::framework::Matrix;
fn main() {
    // use ml_rs::complex_gates::Xor;
    // let mut m = Xor::new(&XOR_TRAIN);
    // m.run()

    // use ml_rs::simple_series;
    // simple_series::simple_model();

    // Regular numeric matrix
    let mut numeric_matrix: Matrix<f64> = Matrix::new(2, 2);

    // Nested matrix
    let nested_matrix: Matrix<Matrix<f64>> = Matrix::new(2, 2);

    // Works with all standard numeric types
    let int_matrix: Matrix<i32> = Matrix::new(2, 2);
    let uint_matrix: Matrix<u64> = Matrix::new(2, 2);
    let float_matrix: Matrix<f32> = Matrix::new(2, 2);

    println!("{:?}", numeric_matrix);
    println!("{:?}", nested_matrix);
    println!("{:?}", int_matrix);
    println!("{:?}", uint_matrix);
    println!("{:?}", float_matrix);
}
