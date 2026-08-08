mod batch;
mod executor;
mod kv;
mod sampling;
mod scheduler;
mod state;
mod types;

#[cfg(any(test, feature = "test-backend"))]
pub mod testing;

pub use batch::{
    ForwardBatch, ForwardBatchError, ForwardKind, ForwardPhase, ForwardSequence, KvSlot, Position,
    QueryRow, SamplingInput, SequenceIndex, TokenId,
};
#[cfg(any(test, feature = "cuda", feature = "test-backend"))]
pub(crate) use executor::ImmediateCompletion;
pub use executor::{
    BatchLoweringError, BatchTicket, CompletionId, ExecutionError, ExecutionOutput, ExecutionStats,
    GeneratedTokens, ModelExecutor,
};
pub use sampling::{HostLogitsSampler, SamplingError};
pub use scheduler::{EngineHandle, EngineSpawnError, RequestStream, SubmitError};
pub use state::{StateArenaId, StateGroupKind, StateGroupSpec, StateSchema, StateSchemaError};
pub use types::{
    FinishReason, GenerateRequest, GenerationEvent, GenerationParams, RequestId, Usage,
};
