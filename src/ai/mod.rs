pub mod batch_inference;
pub mod context_aware;
pub mod distributed;
pub mod ensemble;
pub mod modular_surrogate;
pub mod neural_field;
pub mod shared_batch_service;
pub mod surrogate;
pub mod xdt_export;

// Re-export SurrogateOps and SurrogateOpsBox for use in thermal model
pub use surrogate::{SurrogateManager, SurrogateOps, SurrogateOpsBox};
