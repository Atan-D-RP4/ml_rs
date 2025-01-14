// Usage Example
//
// let matrix_a = Matrix::<f32>::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
// let matrix_b = Matrix::<f32>::from_vec(vec![3.0, 4.0, 3.0, 4.0], 2, 2).unwrap();
// let matrix_c = (matrix_a * matrix_b).unwrap();
// println!("{}", matrix_c);
//
// let matrix_a = matrix![[1, 2], [2, 2]];
// let or_data = matrix![
//     [0.0, 0.0, 0.0],
//     [0.0, 1.0, 1.0],
//     [1.0, 0.0, 1.0],
//     [1.0, 1.0, 1.0],
// ];
// let matrix_b = Matrix::from_vec2(vec![vec![1, 2], vec![3, 4]]).unwrap();
// println!("{}", (matrix_a * matrix_b).unwrap());
// println!("{}", or_data);
//
// let (x, y) = or_data.split_vert(4).unwrap_or_else(|e| {
//     println!("{:?}", e);
//     (Matrix::new(0, 0), Matrix::new(0, 0))
// });
// println!("{}", x);
// println!("{}", y);
//
// let (x, y) = or_data.split_horz(4).unwrap_or_else(|e| {
//     println!("{:?}", e);
//     (Matrix::new(0, 0), Matrix::new(0, 0))
// });
// println!("{}", x);
// println!("{}", y);
//
// let matrix = matrix!(gen 10, 10, || rand::thread_rng().gen_range(0..10));
// println!("{}", matrix);
//
// NOTE: This entire thing is overcomplicated as hell
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug)]
pub enum MatrixError {
    EmptyMatrix,
    EmptyRow,
    InconsistentRowLength,

    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    NonSquareMatrix {
        dimensions: (usize, usize),
    },
    Singular,
    IndexOutOfBounds {
        index: (usize, usize),
        dimensions: (usize, usize),
    },
}

pub trait MatrixElement:
    Clone
    + Default
    + Debug
    + Copy
    + PartialOrd
    + PartialEq
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
}

// Implement for standard numeric types
macro_rules! impl_matrix_element {
    ($($t:ty),*) => {
        $(
            impl MatrixElement for $t {
                fn zero() -> Self { 0 as $t }
                fn one() -> Self { 1 as $t }
                fn abs(self) -> Self { self.abs() }
            }
        )*
    }
}

impl_matrix_element!(f32, f64, i32, i64);

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: MatrixElement> {
    pub data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: MatrixElement> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![T::zero(); rows * cols];
        Matrix { data, rows, cols }
    }

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (rows, cols),
                got: (data.len() / cols, cols),
            });
        }
        Ok(Matrix { data, rows, cols })
    }

    pub fn from_vec2(data: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        let rows = data.len();
        let cols = data[0].len();
        let mut flat_data = Vec::with_capacity(rows * cols);
        for row in data {
            if row.len() != cols {
                return Err(MatrixError::DimensionMismatch {
                    expected: (rows, cols),
                    got: (rows, row.len()),
                });
            }
            flat_data.extend(row);
        }
        Ok(Matrix {
            data: flat_data,
            rows,
            cols,
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /*
        stride = 2
        [0.0, 0.0,| 0.0],
        [0.0, 1.0,| 1.0],
        [1.0, 0.0,| 1.0],
        [1.0, 1.0,| 1.0],
    */
    pub fn split_vert(&self, stride: usize) -> Result<(Self, Self), MatrixError> {
        // Bounds check for stride
        if stride >= self.cols {
            return Err(MatrixError::IndexOutOfBounds {
                index: (0, stride),
                dimensions: (self.rows, self.cols),
            });
        }
        let mut left = Self::new(self.rows, stride);
        let mut right = Self::new(self.rows, self.cols - stride);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j < stride {
                    left[(i, j)] = self[(i, j)].clone();
                } else {
                    right[(i, j - stride)] = self[(i, j)].clone();
                }
            }
        }
        Ok((left, right))
    }

    pub fn split_horz(&self, stride: usize) -> Result<(Self, Self), MatrixError> {
        // Bounds check for stride
        if stride >= self.rows {
            return Err(MatrixError::IndexOutOfBounds {
                index: (stride, 0),
                dimensions: (self.rows, self.cols),
            });
        }

        let mut top = Self::new(stride, self.cols);
        let mut bottom = Self::new(self.rows - stride, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if i < stride {
                    top[(i, j)] = self[(i, j)].clone();
                } else {
                    bottom[(i - stride, j)] = self[(i, j)].clone();
                }
            }
        }
        Ok((top, bottom))
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.cols)
    }

    pub fn iter_cols(&self) -> impl Iterator<Item = Vec<T>> + use<'_, T> {
        (0..self.cols).map(move |i| {
            (0..self.rows)
                .map(move |j| self[(j, i)].clone())
                .collect::<Vec<T>>()
        })
    }
}

// Algebraic Operations
impl<T: MatrixElement> Matrix<T> {
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix[(i, i)] = T::one();
        }
        matrix
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)].clone();
            }
        }
        result
    }

    // Matrix multiplication
    pub fn multiply(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.cols, self.cols),
                got: (other.rows, other.cols),
            });
        }

        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                for k in 0..self.cols {
                    sum = sum + self[(i, k)].clone() * other[(k, j)].clone();
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }

    // Gaussian elimination
    pub fn gaussian_elimination(&self) -> Result<Self, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquareMatrix {
                dimensions: (self.rows, self.cols),
            });
        }

        let mut result = self.clone();
        let n = self.rows;

        for i in 0..n {
            // Find pivot
            let mut max_element = result[(i, i)].clone().abs();
            let mut max_row = i;
            for k in (i + 1)..n {
                let abs_element = result[(k, i)].clone().abs();
                if abs_element > max_element {
                    max_element = abs_element;
                    max_row = k;
                }
            }

            // Check if matrix is singular
            if max_element == T::zero() {
                return Err(MatrixError::Singular);
            }

            // Swap maximum row with current row
            if max_row != i {
                for k in 0..n {
                    let temp = result[(i, k)].clone();
                    result[(i, k)] = result[(max_row, k)].clone();
                    result[(max_row, k)] = temp;
                }
            }

            // Make all rows below this one zero in current column
            for k in (i + 1)..n {
                let c = result[(k, i)].clone() / result[(i, i)].clone();
                for j in 0..n {
                    result[(k, j)] = result[(k, j)].clone() - c.clone() * result[(i, j)].clone();
                }
            }
        }

        Ok(result)
    }

    pub fn determinant(&self) -> Result<T, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquareMatrix {
                dimensions: (self.rows, self.cols),
            });
        }

        let triangular = self.gaussian_elimination()?;
        let mut det = T::one();
        for i in 0..self.rows {
            det = det * triangular[(i, i)].clone();
        }
        Ok(det)
    }

    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquareMatrix {
                dimensions: (self.rows, self.cols),
            });
        }

        let n = self.rows;
        let mut augmented = Self::new(n, 2 * n);

        // Create augmented matrix [A|I]
        for i in 0..n {
            for j in 0..n {
                augmented[(i, j)] = self[(i, j)].clone();
                augmented[(i, j + n)] = if i == j { T::one() } else { T::zero() };
            }
        }

        // Apply Gaussian elimination
        for i in 0..n {
            let pivot = augmented[(i, i)].clone();
            if pivot == T::zero() {
                return Err(MatrixError::Singular);
            }

            // Scale row to make pivot 1
            for j in 0..(2 * n) {
                augmented[(i, j)] = augmented[(i, j)].clone() / pivot.clone();
            }

            // Make all other rows zero in current column
            for k in 0..n {
                if k != i {
                    let factor = augmented[(k, i)].clone();
                    for j in 0..(2 * n) {
                        augmented[(k, j)] =
                            augmented[(k, j)].clone() - factor.clone() * augmented[(i, j)].clone();
                    }
                }
            }
        }

        // Extract right half of augmented matrix
        let mut result = Self::new(n, n);
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] = augmented[(i, j + n)].clone();
            }
        }

        Ok(result)
    }
}

// Implement indexing operations
impl<T: MatrixElement> std::ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl<T: MatrixElement> std::ops::IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

// Implement Dereference trait
impl<T: MatrixElement> std::ops::Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: MatrixElement> std::ops::DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

// OPERATOR OVERLOADS
// Implement matrix addition
impl<T: MatrixElement> Add for Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn add(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = self[(i, j)].clone() + other[(i, j)].clone();
            }
        }
        Ok(result)
    }
}

// Implement matrix subtraction
impl<T: MatrixElement> Sub for Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn sub(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = self[(i, j)].clone() - other[(i, j)].clone();
            }
        }
        Ok(result)
    }
}

// Implement matrix multiplication
impl<T: MatrixElement + PartialOrd> Mul for Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn mul(self, other: Self) -> Self::Output {
        self.multiply(&other)
    }
}

impl<T: MatrixElement> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = MatrixError;

    fn try_from(data: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        if data.is_empty() {
            return Err(MatrixError::EmptyMatrix);
        }

        let rows = data.len();
        let cols = data[0].len();

        if cols == 0 {
            return Err(MatrixError::EmptyRow);
        }

        // Validate all rows have same length
        if !data.iter().all(|row| row.len() == cols) {
            return Err(MatrixError::InconsistentRowLength);
        }

        // Flatten the 2D vector into 1D
        let flat_data: Vec<T> = data.into_iter().flatten().collect();

        Ok(Matrix {
            data: flat_data,
            rows,
            cols,
        })
    }
}

#[macro_export]
macro_rules! matrix {
    // Match empty matrix
    () => {
        compile_error!("Matrix cannot be empty")
    };

    // Match single empty row
    ([]) => {
        compile_error!("Matrix cannot have empty rows")
    };

    // Array initialization syntax: [[expr; cols]; rows]
    ([[$(($init:expr)),+; $cols:expr]; $rows:expr]) => {{
        let data: Vec<Vec<_>> = vec![vec![$($init),+; $cols]; $rows];
        Matrix::try_from(data).expect("Failed to create matrix")
    }};

    // Direct value syntax: [[1, 2], [3, 4]]
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        let data: Vec<Vec<_>> = vec![$(vec![$($x),*]),+];
        Matrix::try_from(data).expect("Failed to create matrix")
    }};

    // Match matrix with rows
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        {
            let temp: Vec<Vec<_>> = vec![
                $(
                    {
                        let row = vec![$($x),*];
                        row
                    }
                ),+
            ];
            Matrix::try_from(temp).expect("Failed to create matrix")
        }
    }};

    // Match a generator expression
    (gen $rows:expr, $cols:expr, $gen:expr) => {{
        {
            let mut data = Vec::with_capacity($rows * $cols);
            for _ in 0..$rows {
                for _ in 0..$cols {
                    data.push($gen());
                }
            }
            Matrix::from_vec(data, $rows, $cols).expect("Failed to create matrix")
        }
    }};
}

impl<T: Debug + MatrixElement> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:?} ", self[(i, j)])?;
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
    fn test_matrix_macro() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9],];
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
    }

    #[test]
    fn test_matrix_addition() {
        let matrix_a = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9],];
        let matrix_b = matrix![[9, 8, 7], [6, 5, 4], [3, 2, 1],];
        let result = matrix_a + matrix_b;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result[(0, 0)], 10);
        assert_eq!(result[(1, 1)], 10);
        assert_eq!(result[(2, 2)], 10);
    }

    #[test]
    fn test_matrix_subtraction() {
        let matrix_a = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9],];
        let matrix_b = matrix![[9, 8, 7], [6, 5, 4], [3, 2, 1],];
        let result = matrix_a - matrix_b;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result[(0, 0)], -8);
        assert_eq!(result[(1, 1)], 0);
        assert_eq!(result[(2, 2)], 8);
    }

    #[test]
    fn test_matrix_multiplication() {
        let matrix_a = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9],];
        let matrix_b = matrix![[9, 8, 7], [6, 5, 4], [3, 2, 1],];
        let result = matrix_a * matrix_b;
        assert!(result.is_ok());
        let result = result.unwrap();
        let expected = matrix![[30, 24, 18], [84, 69, 54], [138, 114, 90],];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(result[(i, j)], expected[(i, j)]);
            }
        }
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9],];
        let result = matrix.transpose();
        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(1, 0)], 2);
        assert_eq!(result[(2, 0)], 3);
        assert_eq!(result[(0, 1)], 4);
        assert_eq!(result[(1, 1)], 5);
        assert_eq!(result[(2, 1)], 6);
        assert_eq!(result[(0, 2)], 7);
        assert_eq!(result[(1, 2)], 8);
        assert_eq!(result[(2, 2)], 9);
    }

    #[test]
    fn test_matrix_split() {
        let matrix = matrix![
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        let (top, bottom) = matrix.split_horz(2).unwrap();
        assert_eq!(top.rows, 2);
        assert_eq!(top.cols, 4);
        assert_eq!(bottom.rows, 2);
        assert_eq!(bottom.cols, 4);

        let (left, right) = matrix.split_vert(2).unwrap();
        assert_eq!(left.rows, 4);
        assert_eq!(left.cols, 2);
        assert_eq!(right.rows, 4);
        assert_eq!(right.cols, 2);
    }
}
