//! Observation and action space descriptors.

/// A continuous tensor or discrete choice space.
#[derive(Clone, Debug, PartialEq)]
pub enum Space {
    /// An `f32` tensor with uniform scalar bounds.
    Box {
        shape: Vec<usize>,
        low: f32,
        high: f32,
    },
    Discrete {
        n: usize,
    },
}

impl Space {
    /// A `[0, 1]`-bounded tensor space.
    pub fn unit_box(shape: Vec<usize>) -> Space {
        Space::Box {
            shape,
            low: 0.0,
            high: 1.0,
        }
    }
}
