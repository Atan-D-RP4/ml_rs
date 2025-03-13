use rand::Rng;
use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

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

// New unified view types
#[derive(Debug)]
pub struct MatrixView<'a, T: MatrixElement> {
    pub elements: &'a [T],
    pub rows: usize,
    pub cols: usize,
    pub offset_row: usize,
    pub offset_col: usize,
}

#[derive(Debug)]
pub struct MatrixViewMut<'a, T: MatrixElement> {
    pub elements: &'a mut [T],
    pub rows: usize,
    pub cols: usize,
    pub offset_row: usize,
    pub offset_col: usize,
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

// Implement Index and IndexMut for the view types
impl<'a, T: MatrixElement> Index<usize> for MatrixView<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < (self.rows + self.offset_row) * (self.cols + self.offset_col));
        let index = index - self.offset_row * self.cols - self.offset_col;
        &self.elements[index]
    }
}

impl<'a, T: MatrixElement> Index<usize> for MatrixViewMut<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < (self.rows + self.offset_row) * (self.cols + self.offset_col));
        let index = index - self.offset_row * self.cols - self.offset_col;
        &self.elements[index]
    }
}

impl<'a, T: MatrixElement> IndexMut<usize> for MatrixViewMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < (self.rows + self.offset_row) * (self.cols + self.offset_col));
        let index = index - self.offset_row * self.cols - self.offset_col;
        &mut self.elements[index]
    }
}

impl<T: MatrixElement> Iterator for Matrix<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements.iter().next().copied()
    }
}

impl<'a, T: MatrixElement> Iterator for MatrixView<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements.iter().enumerate().skip_while(|(i, _)| {
            i < &(self.offset_row * self.cols + self.offset_col)
        }).next().map(|(_, e)| *e)
    }
}

impl<'a, T: MatrixElement> Iterator for MatrixViewMut<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements.iter_mut().enumerate().skip_while(|(i, _)| {
            i < &(self.offset_row * self.cols + self.offset_col)
        }).next().map(|(_, e)| *e)
    }
}
