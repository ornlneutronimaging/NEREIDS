//! NeXus/HDF5 I/O stubs for NEREIDS.

#[derive(Debug)]
pub enum IoError {
    NotImplemented,
}

#[derive(Debug, Default)]
pub struct NexusWriter;

impl NexusWriter {
    pub fn new() -> Self {
        Self
    }

    pub fn write_stub(&self) -> Result<(), IoError> {
        Err(IoError::NotImplemented)
    }
}
