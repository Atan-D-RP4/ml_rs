use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

// Define a trait that captures the requirements for a type that can be stored in our matrix
pub trait MatrixElement: Clone + Default {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
}

// Implement MatrixElement for all number types that satisfy our basic requirements
impl<T> MatrixElement for T
where
    T: Clone
        + Default
        + Debug
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    fn is_zero(&self) -> bool {
        self == &T::default()
    }

    fn is_one(&self) -> bool {
        // This is a simplification - proper implementation would need to check against actual one value
        // for the specific numeric type
        false
    }
}

// Implement MatrixElement for nested matrices
impl<T: MatrixElement> MatrixElement for Matrix<T> {
    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }

    fn is_one(&self) -> bool {
        // A matrix is considered one if it's an identity matrix
        if self.rows != self.cols {
            return false;
        }

        self.data.chunks(self.cols).enumerate().all(|(i, row)| {
            row.iter().enumerate().all(|(j, elem)| {
                if i == j {
                    elem.is_one()
                } else {
                    elem.is_zero()
                }
            })
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Matrix<T: MatrixElement> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: MatrixElement> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![T::default(); rows * cols];
        Matrix { data, rows, cols }
    }

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Matrix { data, rows, cols })
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            self.data.get(row * self.cols + col)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            self.data.get_mut(row * self.cols + col)
        } else {
            None
        }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}
