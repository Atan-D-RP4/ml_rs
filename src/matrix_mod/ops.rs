use super::error::MatrixError;
use super::types::{Matrix, MatrixElement};

impl<T: MatrixElement> Matrix<T> {
    // NOTE: DONE
    pub fn randomize(&mut self, low: T, high: T) {
        for element in &mut self.elements {
            *element = T::random(low, high);
        }
    }

    // NOTE: DONE
    pub fn add(&mut self, other: &Matrix<T>) -> Result<&mut Self, MatrixError> {
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
    pub fn sub(&mut self, other: &Matrix<T>) -> Result<&mut Self, MatrixError> {
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

    pub fn transpose(&mut self) -> Result<&mut Self, MatrixError> {
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
