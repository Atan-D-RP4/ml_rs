use crate::matrix::MatrixError;
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum NNError {
    ArchitectureError { msg: String, details: Option<String> },
    InputError { msg: String, expected_size: usize, actual_size: usize },
    DataSetError { msg: String, stride: usize, total_columns: usize },
    MatrixError { msg: String, operation: String },
    TrainingError { msg: String, cost: Option<f32> },
}

impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NNError::ArchitectureError { msg, details } => {
                write!(f, "Architecture Error: {}", msg)?;
                if let Some(detail) = details {
                    write!(f, " ({})", detail)?;
                }
                Ok(())
            }
            NNError::InputError {
                msg,
                expected_size,
                actual_size,
            } => {
                write!(f, "Input Error: {} (expected {}, got {})", msg, expected_size, actual_size)
            }
            NNError::DataSetError { msg, stride, total_columns } => {
                write!(f, "DataSet Error: {} (stride: {}, total columns: {})", msg, stride, total_columns)
            }
            NNError::MatrixError { msg, operation } => {
                write!(f, "Matrix Operation Error: {} during {}", msg, operation)
            }
            NNError::TrainingError { msg, cost } => {
                write!(f, "Training Error: {}", msg)?;
                if let Some(c) = cost {
                    write!(f, " (current cost: {})", c)?;
                }
                Ok(())
            }
        }
    }
}

impl Error for NNError {}

impl From<MatrixError> for NNError {
    fn from(error: MatrixError) -> Self {
        NNError::MatrixError {
            msg: error.to_string(),
            operation: "Matrix Operation".to_string(),
        }
    }
}
