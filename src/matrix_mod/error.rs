use std::{error::Error, fmt};

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
