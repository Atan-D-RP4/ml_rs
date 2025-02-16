use crate::matrix::Matrix;
use crate::nn::error::NNError;

#[derive(Debug, Default)]
pub struct DataSet {
    pub data: Matrix<f32>,
    pub stride: usize,
}

impl DataSet {
    pub fn new(data: Matrix<f32>, stride: usize) -> Result<Self, NNError> {
        if stride >= data.cols {
            return Err(NNError::DataSetError {
                msg: "Stride cannot be greater than the total columns".to_string(),
                stride,
                total_columns: data.cols,
            });
        }
        Ok(Self { data, stride })
    }

    pub fn inputs(&self) -> Vec<Matrix<f32>> {
        let mut inputs = Vec::new();
        for i in 0..self.data.rows {
            let mut inp_row = Vec::new();
            for j in 0..self.stride {
                inp_row.push(self.data[(i, j)]);
            }
            inputs.push(Matrix::from_vec2d(vec![inp_row]).unwrap());
        }
        inputs
    }

    pub fn inputs_as_matrix(&self) -> Matrix<f32> {
        let inputs = self.inputs();
        inputs.iter().fold(Matrix::new(0, inputs[0].cols), |mut acc, x| {
            acc.vstack(x).expect("Matrix stacking failed");
            acc
        })
    }

    pub fn targets(&self) -> Vec<Matrix<f32>> {
        let mut targets = Vec::new();
        for i in 0..self.data.rows {
            let mut target_row = Vec::new();
            for j in self.stride..self.data.cols {
                target_row.push(self.data[(i, j)]);
            }
            targets.push(Matrix::from_vec2d(vec![target_row]).unwrap());
        }
        targets
    }

    pub fn targets_as_matrix(&self) -> Matrix<f32> {
        let targets = self.targets();
        targets.iter().fold(Matrix::new(0, targets[0].cols), |mut acc, x| {
            acc.vstack(x).expect("Matrix stacking failed");
            acc
        })
    }
}

impl std::fmt::Display for DataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DataSet {{\n")?;
        // [[inputs], [targets]] based on stride
        self.data.elements.iter().enumerate().for_each(|(i, val)| {
            if i % self.data.cols == 0 {
                write!(f, "  [").unwrap();
            }
            write!(f, "{}, ", val).unwrap();
            if (i + 1) % self.data.cols == self.stride {
                write!(f, "|").unwrap();
            }
            if (i + 1) % self.data.cols == 0 {
                write!(f, "]\n").unwrap();
            }
        });

        write!(f, "}}")
    }
}
