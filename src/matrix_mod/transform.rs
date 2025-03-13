use super::error::MatrixError;
use super::types::{Matrix, MatrixElement, MatrixView, MatrixViewMut};

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
            offset_col: 0,
            offset_row: index,
        }
    }

    pub fn row_mut(&mut self, index: usize) -> MatrixViewMut<T> {
        assert!(index < self.rows);
        MatrixViewMut {
            elements: &mut self.elements[index * self.cols..(index + 1) * self.cols],
            rows: 1,
            cols: self.cols,
            offset_col: 0,
            offset_row: index,
        }
    }

    pub fn col(&self, index: usize) -> MatrixView<T> {
        assert!(index < self.cols);
        MatrixView {
            elements: self.elements.as_slice(),
            rows: self.rows,
            cols: 1,
            offset_row: 0,
            offset_col: index,
        }
    }

    pub fn col_mut(&mut self, index: usize) -> MatrixViewMut<T> {
        assert!(index < self.cols);
        MatrixViewMut {
            elements: self.elements.as_mut_slice(),
            rows: self.rows,
            cols: 1,
            offset_row: 0,
            offset_col: index,
        }
    }
}
