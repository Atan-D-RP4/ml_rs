use rand::Rng;
use std::fmt::Debug;
use std::ops::Div;
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

// MatrixElement trait and its implementations remain the same
pub trait MatrixElement:
    Default
    + Display
    + Debug
    + Clone
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + PartialEq
    + PartialOrd
{
    fn random(low: Self, high: Self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
}

macro_rules! impl_matrix_element {
    ($($t:ty),*) => {
        $(
            impl MatrixElement for $t {
                fn random(low: Self, high: Self) -> Self {
                    let mut rng = rand::thread_rng();
                    match std::any::type_name::<$t>() {
                        "f32" => rng.gen::<f32>() as $t * (high - low) + low,
                        "f64" => rng.gen::<f64>() as $t * (high - low) + low,
                        _ => rng.gen_range(low..=high),
                    }
                }
                fn zero() -> Self{
                    Self::default()
                }

                fn one() -> Self {
                    1 as Self
                }

                fn is_zero(&self) -> bool {
                    *self == Self::zero()
                }
            }
        )*
    }
}

impl_matrix_element!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

// Main Matrix struct remains mostly the same
#[derive(Debug, Clone)]
pub struct Matrix<T: MatrixElement> {
    pub rows: usize,
    pub cols: usize,
    pub elements: Vec<T>,
}

impl<T: MatrixElement> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows && col < self.cols);
        &self.elements[row * self.cols + col]
    }
}

impl<T: MatrixElement> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows && col < self.cols);
        &mut self.elements[row * self.cols + col]
    }
}

// Base Matrix implementation remains the same, just showing relevant methods
impl<T: MatrixElement> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elements: vec![T::default(); rows * cols],
        }
    }

    pub fn from_vec2d(elements: Vec<Vec<T>>) -> Self {
        let rows = elements.len();
        let cols = elements[0].len();
        let mut flat_elements = Vec::with_capacity(rows * cols);
        for row in elements {
            flat_elements.extend(row);
        }
        Self {
            rows,
            cols,
            elements: flat_elements,
        }
    }

    pub fn from_slice(rows: usize, cols: usize, elements: &[T]) -> Self {
        assert_eq!(elements.len(), rows * cols);
        Self {
            rows,
            cols,
            elements: elements.to_vec(),
        }
    }

    pub fn fill(&mut self, value: T) {
        self.elements.fill(value);
    }

    pub fn randomize(&mut self, low: T, high: T) {
        for element in &mut self.elements {
            *element = T::random(low, high);
        }
    }

    pub fn add(&mut self, other: &Matrix<T>) {
        // Handle broadcasting for 1xN matrix being added to MxN matrix
        if other.rows == 1 {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    self[(i, j)] += other[(0, j)];
                }
            }
            return;
        }

        // Regular case - matrices must have same dimensions
        assert_eq!(
            self.rows, other.rows,
            "Matrix dimensions must match for addition"
        );
        assert_eq!(
            self.cols, other.cols,
            "Matrix dimensions must match for addition"
        );

        for i in 0..self.elements.len() {
            self.elements[i] += other.elements[i];
        }
    }

    pub fn sub(&mut self, other: &Matrix<T>) {
        if other.rows == 1 && other.cols == 1 {
            for element in &mut self.elements {
                *element -= other[(0, 0)];
            }
            return;
        }
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        for i in 0..self.elements.len() {
            self.elements[i] -= other.elements[i];
        }
    }

    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    pub fn row(&self, index: usize) -> MatrixView<T> {
        assert!(index < self.rows);
        MatrixView {
            elements: &self.elements[index * self.cols..(index + 1) * self.cols],
            rows: 1,
            cols: self.cols,
            stride: 1,
            offset: 0,
        }
    }

    pub fn row_mut(&mut self, index: usize) -> MatrixViewMut<T> {
        assert!(index < self.rows);
        let cols = self.cols;
        MatrixViewMut {
            elements: &mut self.elements[index * cols..(index + 1) * cols],
            rows: 1,
            cols,
            stride: 1,
            offset: 0,
        }
    }

    pub fn col(&self, index: usize) -> MatrixView<T> {
        assert!(index < self.cols);
        MatrixView {
            elements: self.elements.as_slice(),
            rows: self.rows,
            cols: 1,
            stride: self.cols,
            offset: index,
        }
    }

    pub fn col_mut(&mut self, index: usize) -> MatrixViewMut<T> {
        assert!(index < self.cols);
        MatrixViewMut {
            elements: self.elements.as_mut_slice(),
            rows: self.rows,
            cols: 1,
            stride: self.cols,
            offset: index,
        }
    }
}

impl<T: MatrixElement> Matrix<T> {
    pub fn transpose(&mut self) {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed[(j, i)] = self[(i, j)];
            }
        }
        *self = transposed;
    }

    pub fn minor(&self, row: usize, col: usize) -> Matrix<T> {
        assert!(row < self.rows && col < self.cols);
        let mut minor = Matrix::new(self.rows - 1, self.cols - 1);
        let mut minor_row = 0;
        let mut minor_col;

        for i in 0..self.rows {
            if i == row {
                continue;
            }
            minor_col = 0;
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                minor[(minor_row, minor_col)] = self[(i, j)];
                minor_col += 1;
            }
            minor_row += 1;
        }
        minor
    }

    pub fn determinant(&self) -> T {
        assert_eq!(
            self.rows, self.cols,
            "Matrix must be square to compute determinant"
        );

        match self.rows {
            0 => T::zero(),
            1 => self[(0, 0)],
            2 => {
                // For 2x2 matrix, compute ad - bc directly
                let prod1 = self[(0, 0)] * self[(1, 1)];
                let prod2 = self[(0, 1)] * self[(1, 0)];
                prod1.sub(prod2)
            }
            _ => {
                // For larger matrices, use Laplace expansion along first row
                let mut result = T::zero();
                let mut add_next = true;

                for j in 0..self.cols {
                    let minor_det = self.minor(0, j).determinant();
                    let term = self[(0, j)] * minor_det;

                    if add_next {
                        result = result + term;
                    } else {
                        result = result.sub(term);
                    }
                    add_next = !add_next;
                }
                result
            }
        }
    }
}

// Helper method to compute determinant with explicit sign handling
impl<T: MatrixElement> Matrix<T> {
    pub fn adjugate(&self) -> Matrix<T> {
        assert_eq!(
            self.rows, self.cols,
            "Matrix must be square to compute adjugate"
        );
        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                // Compute cofactor
                let minor = self.minor(i, j);
                let cofactor = minor.determinant();
                // Apply checkerboard pattern for sign
                result[(j, i)] = if (i + j) % 2 == 0 {
                    cofactor
                } else {
                    T::zero().sub(cofactor)
                };
            }
        }
        result
    }

    pub fn inverse(&self) -> Option<Matrix<T>> {
        assert_eq!(
            self.rows, self.cols,
            "Matrix must be square to compute inverse"
        );

        let det = self.determinant();
        if det.is_zero() {
            return None; // Matrix is not invertible
        }

        // Compute adjugate matrix
        let adj = self.adjugate();

        // Divide adjugate by determinant
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = adj[(i, j)].div(det);
            }
        }

        Some(result)
    }

    // Helper method to create identity matrix
    pub fn identity(size: usize) -> Self {
        let mut matrix = Matrix::new(size, size);
        for i in 0..size {
            matrix[(i, i)] = T::one();
        }
        matrix
    }

    // Helper method to verify if a matrix is the identity matrix
    pub fn is_identity(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                let expected = if i == j { T::one() } else { T::zero() };
                if self[(i, j)] != expected {
                    return false;
                }
            }
        }
        true
    }

    // Method to validate inverse calculation
    pub fn verify_inverse(&self, inverse: &Matrix<T>) -> bool {
        let product = self.dot(inverse);
        product.is_identity()
    }

    pub fn determinant_with_cofactors(&self) -> (T, bool) {
        assert_eq!(
            self.rows, self.cols,
            "Matrix must be square to compute determinant"
        );

        match self.rows {
            0 => (T::zero(), true),
            1 => (self[(0, 0)], true),
            2 => {
                let prod1 = self[(0, 0)] * self[(1, 1)];
                let prod2 = self[(0, 1)] * self[(1, 0)];
                if prod1 >= prod2 {
                    (prod1.sub(prod2), true)
                } else {
                    (prod2.sub(prod1), false)
                }
            }
            _ => {
                let mut result = T::zero();
                let mut is_positive = true;

                for j in 0..self.cols {
                    let (minor_det, minor_sign) = self.minor(0, j).determinant_with_cofactors();
                    let term = self[(0, j)] * minor_det;
                    let add_this_term = (j % 2 == 0) == minor_sign;

                    if add_this_term == is_positive {
                        result = result + term;
                    } else {
                        if result >= term {
                            result = result.sub(term);
                        } else {
                            result = term.sub(result);
                            is_positive = !is_positive;
                        }
                    }
                }
                (result, is_positive)
            }
        }
    }
}

// New unified view types
#[derive(Debug)]
pub struct MatrixView<'a, T: MatrixElement> {
    elements: &'a [T],
    rows: usize,
    cols: usize,
    stride: usize,
    offset: usize,
}

#[derive(Debug)]
pub struct MatrixViewMut<'a, T: MatrixElement> {
    elements: &'a mut [T],
    rows: usize,
    cols: usize,
    stride: usize,
    offset: usize,
}

impl<'a, T: MatrixElement> MatrixView<'a, T> {
    pub fn to_matrix(&self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                elements.push(self.elements[row * self.stride + col + self.offset]);
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            elements,
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.rows * self.cols {
            let row = index / self.cols;
            let col = index % self.cols;
            Some(&self.elements[row * self.stride + col + self.offset])
        } else {
            None
        }
    }
}

impl<'a, T: MatrixElement> MatrixViewMut<'a, T> {
    pub fn to_matrix(&self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                elements.push(self.elements[row * self.stride + col + self.offset]);
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            elements,
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.rows * self.cols {
            let row = index / self.cols;
            let col = index % self.cols;
            Some(&self.elements[row * self.stride + col + self.offset])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.rows * self.cols {
            let row = index / self.cols;
            let col = index % self.cols;
            Some(&mut self.elements[row * self.stride + col + self.offset])
        } else {
            None
        }
    }

    pub fn fill(&mut self, value: T) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.elements[row * self.stride + col + self.offset] = value;
            }
        }
    }

    pub fn copy_from(&mut self, other: &MatrixView<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                self.elements[row * self.stride + col + self.offset] =
                    other.elements[row * other.stride + col + other.offset];
            }
        }
    }
}

// Implement Index and IndexMut for the view types
impl<'a, T: MatrixElement> Index<usize> for MatrixView<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<'a, T: MatrixElement> Index<usize> for MatrixViewMut<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<'a, T: MatrixElement> IndexMut<usize> for MatrixViewMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of bounds")
    }
}

impl<T: MatrixElement> Matrix<T> {
    pub fn apply_fn<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        for element in &mut self.elements {
            *element = f(*element);
        }
    }

    pub fn apply_with_matrix<F>(&mut self, other: &Matrix<T>, f: F)
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        for i in 0..self.elements.len() {
            self.elements[i] = f(self.elements[i], other.elements[i]);
        }
    }
}

impl<T: MatrixElement> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:.2} ", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matrix_operations() {
        let mut m1 = Matrix::new(2, 3);
        m1.fill(1.0);
        let mut m2 = Matrix::new(3, 2);
        m2.fill(2.0);

        let result = m1.dot(&m2);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result[(0, 0)], 6.0);
    }

    #[test]
    fn test_matrix_view() {
        let mut m = Matrix::new(3, 3);
        m.fill(1.0);
        let view = m.row(1);
        assert_eq!(view[0], 1.0);
    }

    #[test]
    fn test_matrix_view_mut() {
        let mut m = Matrix::new(3, 3);
        m.fill(1.0);
        let mut view = m.row_mut(1);
        view.fill(2.0);
        assert_eq!(view[0], 2.0);
    }

    #[test]
    fn test_matrix_det() {
        let m = Matrix::from_vec2d(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
        println!("Matrix:\n{}", m);
        assert_eq!(m.determinant(), 0);
    }

    #[test]
    fn test_matrix_cofactor() {
        let m = Matrix::from_vec2d(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
        println!("Matrix:\n{}", m);
        let (det, sign) = m.determinant_with_cofactors();
        assert_eq!(det, 0);
        assert_eq!(sign, true);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::from_vec2d(vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ]);
        m.transpose();
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_matrix_add() {
        let mut m1 = Matrix::new(2, 2);
        m1.fill(1.0);
        let mut m2 = Matrix::new(2, 2);
        m2.fill(2.0);

        m1.add(&m2);
        assert_eq!(m1[(0, 0)], 3.0);
    }

    #[test]
    fn test_matrix_dot() {
        let mut m1 = Matrix::new(2, 3);
        m1.fill(1.0);
        let mut m2 = Matrix::new(3, 2);
        m2.fill(2.0);

        let result = m1.dot(&m2);
        assert_eq!(result[(0, 0)], 6.0);
    }
}
