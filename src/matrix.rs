use rand::Rng;
use std::error::Error;
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

#[derive(Debug)]
pub enum MatrixError {
    DimensionMismatch(String),
    IndexOutOfBounds(String),
    InvalidOperation(String),
    SingularMatrix(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch(msg) => write!(f, "Dimension Mismatch: {}", msg),
            MatrixError::IndexOutOfBounds(msg) => write!(f, "Index Out of Bounds: {}", msg),
            MatrixError::InvalidOperation(msg) => write!(f, "Invalid Operation: {}", msg),
            MatrixError::SingularMatrix(msg) => write!(f, "Singular Matrix: {}", msg),
        }
    }
}

impl Error for MatrixError {}

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
                        "f32" | "f64" => rng.gen::<$t>() * (high - low) + low,
                        _ => rng.gen_range(low..=high),
                    }
                }

                fn zero() -> Self {
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

#[derive(Debug, Clone, Default)]
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

// Memory Operations
impl<T: MatrixElement> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elements: vec![T::default(); rows * cols],
        }
    }

    pub fn from_vec2d(elements: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        let rows = elements.len();
        if rows == 0 {
            return Ok(Matrix::new(0, 0));
        }

        let cols = elements[0].len();
        if elements.iter().any(|row| row.len() != cols) {
            return Err(MatrixError::DimensionMismatch(
                "All rows must have the same number of elements".to_string(),
            ));
        }

        let mut flat_elements = Vec::with_capacity(rows * cols);
        for row in elements {
            flat_elements.extend(row);
        }

        let mut result = Matrix::new(rows, cols);
        result.elements = flat_elements;
        Ok(result)
    }

    // NOTE: DONE
    pub fn from_slice(rows: usize, cols: usize, elements: &[T]) -> Result<Self, MatrixError> {
        if elements.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Slice length {} does not match dimensions {} x {}",
                elements.len(),
                rows,
                cols
            )));
        }
        Ok(Self {
            rows,
            cols,
            elements: elements.to_vec(),
        })
    }

    // NOTE: DONE
    pub fn to_slice(&self) -> &[T] {
        &self.elements
    }

    pub fn to_vec2d(&self) -> Vec<Vec<T>> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(self[(i, j)]);
            }
            result.push(row);
        }
        result
    }

    pub fn fill(&mut self, value: T) {
        self.elements.fill(value);
    }

    pub fn vstack(&mut self, other: &Matrix<T>) -> Result<(), MatrixError> {
        if self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot vertically stack matrices with different column counts ({} vs {})",
                self.cols, other.cols
            )));
        }
        self.elements.extend_from_slice(&other.elements);
        self.rows += other.rows;
        Ok(())
    }

    pub fn hstack(&mut self, other: &Matrix<T>) -> Result<(), MatrixError> {
        if self.rows != other.rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot horizontally stack matrices with different row counts ({} vs {})",
                self.rows, other.rows
            )));
        }
        let mut new_elements = Vec::with_capacity(self.elements.len() + other.elements.len());
        for i in 0..self.rows {
            new_elements.extend_from_slice(&self.elements[i * self.cols..(i + 1) * self.cols]);
            new_elements.extend_from_slice(&other.elements[i * other.cols..(i + 1) * other.cols]);
        }
        self.elements = new_elements;
        self.cols += other.cols;
        Ok(())
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

impl<T: MatrixElement> Iterator for Matrix<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements.pop()
    }
}

// Linear Algebra Operations
impl<T: MatrixElement> Matrix<T> {
    // NOTE: DONE
    pub fn randomize(&mut self, low: T, high: T) {
        for element in &mut self.elements {
            *element = T::random(low, high);
        }
    }

    // NOTE: DONE
    pub fn add(&mut self, other: &Matrix<T>) -> Result<&Self, MatrixError> {
        if other.rows == 1 {
            if other.cols != self.cols {
                return Err(MatrixError::DimensionMismatch(
                    "Broadcasting matrix must have same number of columns".to_string(),
                ));
            }
            for i in 0..self.rows {
                for j in 0..self.cols {
                    self[(i, j)] += other[(0, j)];
                }
            }
            return Ok(self);
        }

        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot add matrices of size {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        for i in 0..self.elements.len() {
            self.elements[i] += other.elements[i];
        }
        return Ok(self);
    }

    // NOTE: DONE
    pub fn sub(&mut self, other: &Matrix<T>) -> Result<&Self, MatrixError> {
        if other.rows == 1 && other.cols == 1 {
            let scalar = other[(0, 0)];
            for element in &mut self.elements {
                *element -= scalar;
            }
            return Ok(self);
        }

        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot subtract matrices of size {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        for i in 0..self.elements.len() {
            self.elements[i] -= other.elements[i];
        }
        Ok(self)
    }

    // NOTE: DONE
    pub fn dot(&self, other: &Matrix<T>) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Cannot multiply matrices of size {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                for k in 0..self.cols {
                    sum = sum + self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }

    pub fn transpose(&mut self) -> Result<&Self, MatrixError> {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed[(j, i)] = self[(i, j)];
            }
        }
        *self = transposed;
        Ok(self)
    }

    // NOTE: DONE
    pub fn minor(&self, row: usize, col: usize) -> Result<Self, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "Cannot compute minor for position ({}, {}) in {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

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
        Ok(minor)
    }

    // NOTE: DONE
    pub fn determinant(&self) -> Result<T, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::InvalidOperation(
                "Cannot compute determinant of non-square matrix".to_string(),
            ));
        }

        match self.rows {
            0 => Ok(T::zero()),
            1 => Ok(self[(0, 0)]),
            2 => {
                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(1, 0)];
                let d = self[(1, 1)];
                Ok(a * d - b * c)
            }
            _ => {
                let mut result = T::zero();
                let mut add_next = true;

                for j in 0..self.cols {
                    let minor_det = self.minor(0, j)?.determinant()?;
                    let term = self[(0, j)] * minor_det;

                    if add_next {
                        result = result + term;
                    } else {
                        result = result - term;
                    }
                    add_next = !add_next;
                }
                Ok(result)
            }
        }
    }
}

// Helper method to compute determinant with explicit sign handling
impl<T: MatrixElement> Matrix<T> {
    // NOTE: DONE
    pub fn adjugate(&self) -> Result<Self, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::InvalidOperation(
                "Cannot compute adjugate of non-square matrix".to_string(),
            ));
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let minor = self.minor(i, j)?;
                let cofactor = minor.determinant()?;
                result[(j, i)] = if (i + j) % 2 == 0 {
                    cofactor
                } else {
                    T::zero() - cofactor
                };
            }
        }
        Ok(result)
    }

    // NOTE: DONE
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::InvalidOperation(
                "Cannot compute inverse of non-square matrix".to_string(),
            ));
        }

        let det = self.determinant()?;
        if det.is_zero() {
            return Err(MatrixError::SingularMatrix(
                "Matrix is singular (determinant is zero)".to_string(),
            ));
        }

        let adj = self.adjugate()?;
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = adj[(i, j)] / det;
            }
        }
        Ok(result)
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

    // NOTE: DONE
    // Method to validate inverse calculation
    pub fn verify_inverse(&self, inverse: &Matrix<T>) -> Result<bool, MatrixError> {
        let product = self.dot(inverse)?;
        Ok(product.is_identity())
    }

    // NOTE: DONE
    pub fn determinant_with_cofactors(&self) -> Result<(T, bool), MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::InvalidOperation(
                "Matrix must be square to compute determinant".to_string(),
            ));
        }

        match self.rows {
            0 => Ok((T::zero(), true)),
            1 => Ok((self[(0, 0)], true)),
            2 => {
                let prod1 = self[(0, 0)] * self[(1, 1)];
                let prod2 = self[(0, 1)] * self[(1, 0)];
                if prod1 >= prod2 {
                    Ok((prod1.sub(prod2), true))
                } else {
                    Ok((prod2.sub(prod1), false))
                }
            }
            _ => {
                let mut result = T::zero();
                let mut is_positive = true;

                for j in 0..self.cols {
                    let (minor_det, minor_sign) = self.minor(0, j)?.determinant_with_cofactors()?;
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
                let x = Ok((result, is_positive));
                x
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

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        assert!(row_start + rows <= self.rows);
        assert!(col_start + cols <= self.cols);

        Self {
            elements: self.elements,
            rows,
            cols,
            stride: self.stride,
            offset: self.offset + row_start * self.stride + col_start,
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

    pub fn submatrix_mut<'b: 'a>(
        &'b mut self,
        row_start: usize,
        col_start: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, MatrixError> {
        // assert!(row_start + rows <= self.rows);
        // assert!(col_start + cols <= self.cols);

        if row_start + rows > self.rows {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "Row index out of bounds: {} + {} > {}",
                row_start, rows, self.rows
            )));
        }

        if col_start + cols > self.cols {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "Column index out of bounds: {} + {} > {}",
                col_start, cols, self.cols
            )));
        }

        Ok(Self {
            elements: self.elements,
            rows,
            cols,
            stride: self.stride,
            offset: self.offset + row_start * self.stride + col_start,
        })
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
        F: Fn(T, &Matrix<T>) -> T,
    {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        for i in 0..self.elements.len() {
            self.elements[i] = f(self.elements[i], other);
        }
    }
}

impl<T: MatrixElement> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                write!(f, "{:.2}", self[(i, j)])?;
                if j < self.cols - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "]")?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    fn assert_float_eq(a: f64, b: f64) {
        assert!((a - b).abs() < EPSILON, "{} != {}", a, b);
    }

    fn assert_matrix_eq(a: &Matrix<f64>, b: &Matrix<f64>) {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                assert_float_eq(a[(i, j)], b[(i, j)]);
            }
        }
    }

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::<f64>::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.elements.len(), 6);
    }

    #[test]
    fn test_from_vec2d() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m = Matrix::from_vec2d(data).unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_float_eq(m[(0, 0)], 1.0);
        assert_float_eq(m[(0, 1)], 2.0);
        assert_float_eq(m[(1, 0)], 3.0);
        assert_float_eq(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = Matrix::from_slice(2, 2, &data).unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_float_eq(m[(0, 0)], 1.0);
        assert_float_eq(m[(0, 1)], 2.0);
        assert_float_eq(m[(1, 0)], 3.0);
        assert_float_eq(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_indexing() {
        let mut m = Matrix::<f64>::new(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;

        assert_float_eq(m[(0, 0)], 1.0);
        assert_float_eq(m[(0, 1)], 2.0);
        assert_float_eq(m[(1, 0)], 3.0);
        assert_float_eq(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_addition() {
        let mut m1 = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let m2 = Matrix::from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
        m1.add(&m2).unwrap();

        assert_float_eq(m1[(0, 0)], 6.0);
        assert_float_eq(m1[(0, 1)], 8.0);
        assert_float_eq(m1[(1, 0)], 10.0);
        assert_float_eq(m1[(1, 1)], 12.0);
    }

    #[test]
    fn test_matrix_subtraction() {
        let mut m1 = Matrix::from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
        let m2 = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        m1.sub(&m2).unwrap();

        assert_float_eq(m1[(0, 0)], 4.0);
        assert_float_eq(m1[(0, 1)], 4.0);
        assert_float_eq(m1[(1, 0)], 4.0);
        assert_float_eq(m1[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let m2 = Matrix::from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
        let result = m1.dot(&m2).unwrap();

        assert_float_eq(result[(0, 0)], 19.0);
        assert_float_eq(result[(0, 1)], 22.0);
        assert_float_eq(result[(1, 0)], 43.0);
        assert_float_eq(result[(1, 1)], 50.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        m.transpose().unwrap();

        assert_float_eq(m[(0, 0)], 1.0);
        assert_float_eq(m[(0, 1)], 3.0);
        assert_float_eq(m[(1, 0)], 2.0);
        assert_float_eq(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let det = m.determinant().unwrap();
        assert_float_eq(det, -2.0);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let inv = m.inverse().unwrap();
        let expected = Matrix::from_vec2d(vec![vec![-2.0, 1.0], vec![1.5, -0.5]]).unwrap();
        assert_matrix_eq(&inv, &expected);
    }

    #[test]
    fn test_matrix_views() {
        let mut m = Matrix::from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        // Test row view
        let row = m.row(1);
        assert_eq!(row.rows, 1);
        assert_eq!(row.cols, 3);
        assert_float_eq(row[0], 4.0);
        assert_float_eq(row[1], 5.0);
        assert_float_eq(row[2], 6.0);

        // Test column view
        let col = m.col(1);
        assert_eq!(col.rows, 3);
        assert_eq!(col.cols, 1);
        assert_float_eq(col[0], 2.0);
        assert_float_eq(col[1], 5.0);
        assert_float_eq(col[2], 8.0);

        // Test mutable views
        let mut row_mut = m.row_mut(1);
        row_mut.fill(10.0);
        assert_float_eq(m[(1, 0)], 10.0);
        assert_float_eq(m[(1, 1)], 10.0);
        assert_float_eq(m[(1, 2)], 10.0);
    }

    #[test]
    fn test_matrix_stack_operations() {
        let mut m1 = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let m2 = Matrix::from_vec2d(vec![vec![5.0, 6.0]]).unwrap();

        // Test vertical stack
        m1.vstack(&m2).unwrap();
        assert_eq!(m1.rows, 3);
        assert_eq!(m1.cols, 2);
        assert_float_eq(m1[(2, 0)], 5.0);
        assert_float_eq(m1[(2, 1)], 6.0);

        // Test horizontal stack
        let mut m3 = Matrix::from_vec2d(vec![vec![1.0], vec![2.0]]).unwrap();
        let m4 = Matrix::from_vec2d(vec![vec![3.0], vec![4.0]]).unwrap();
        m3.hstack(&m4).unwrap();
        assert_eq!(m3.rows, 2);
        assert_eq!(m3.cols, 2);
        assert_float_eq(m3[(0, 1)], 3.0);
        assert_float_eq(m3[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::<f64>::identity(3);
        assert!(m.is_identity());

        let mut non_identity = Matrix::<f64>::identity(3);
        non_identity[(0, 1)] = 1.0;
        assert!(!non_identity.is_identity());
    }

    #[test]
    fn test_matrix_randomize() {
        let mut m = Matrix::<f64>::new(3, 3);
        m.randomize(0.0, 1.0);

        // Check if all elements are within range
        for i in 0..m.rows {
            for j in 0..m.cols {
                assert!(m[(i, j)] >= 0.0 && m[(i, j)] <= 1.0);
            }
        }
    }

    #[test]
    fn test_error_handling() {
        // Test dimension mismatch in addition
        let mut m1 = Matrix::from_vec2d(vec![vec![1.0, 2.0]]).unwrap();
        let m2 = Matrix::from_vec2d(vec![vec![1.0]]).unwrap();
        assert!(m1.add(&m2).is_err());

        // Test invalid inverse
        let singular = Matrix::from_vec2d(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).unwrap();
        assert!(singular.inverse().is_err());

        // Test out of bounds indexing
        let m = Matrix::<f64>::new(2, 2);
        let result = std::panic::catch_unwind(|| m[(3, 3)]);
        assert!(result.is_err());
    }
}
