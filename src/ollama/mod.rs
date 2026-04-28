pub mod client;
pub mod mock;
pub mod real;
pub mod types;

pub use client::{OllamaClient};
pub use mock::MockOllamaClient;
pub use real::RealOllamaClient;
pub use types::*;
