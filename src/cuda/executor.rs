use std::sync::Arc;

use crate::{
    engine::{
        BatchTicket, CompletionId, ExecutionError, ExecutionOutput, ExecutionStats, ForwardBatch,
        GeneratedTokens, HostLogitsSampler, ImmediateCompletion, ModelExecutor, StateSchema,
        TokenId,
    },
    model::Model,
};

use super::batch::CudaBatch;

/// Architecture-specific computation called once for a whole lowered batch.
/// CUDA lifecycle, completion, sampling, and engine protocol stay outside it.
pub(crate) trait ModelProgram: 'static {
    fn model(&self) -> Arc<dyn Model>;

    fn state_schema(&self) -> &StateSchema;

    fn execute(&mut self, batch: &CudaBatch) -> Result<ProgramOutput, ExecutionError>;

    fn take_execution_stats(&mut self) -> ExecutionStats {
        ExecutionStats::default()
    }
}

pub(crate) enum ProgramOutput {
    None,
    Tokens(Vec<TokenId>),
    HostLogits { values: Vec<f32>, vocab_size: usize },
}

/// Generic synchronous CUDA executor. Its ticket protocol deliberately matches
/// a future event-backed implementation; only the completion storage changes
/// when model programs become asynchronously enqueued.
pub(crate) struct CudaExecutor<P> {
    program: P,
    batch: CudaBatch,
    completions: ImmediateCompletion,
    host_sampler: HostLogitsSampler,
}

impl<P> CudaExecutor<P> {
    pub fn new(program: P) -> Self {
        Self {
            program,
            batch: CudaBatch::default(),
            completions: ImmediateCompletion::default(),
            host_sampler: HostLogitsSampler,
        }
    }
}

impl<P: ModelProgram> ModelExecutor for CudaExecutor<P> {
    fn model(&self) -> Arc<dyn Model> {
        self.program.model()
    }

    fn state_schema(&self) -> &StateSchema {
        self.program.state_schema()
    }

    fn submit(&mut self, batch: &ForwardBatch) -> Result<BatchTicket, ExecutionError> {
        self.completions.ensure_available()?;
        if batch.arena_id() != self.program.state_schema().arena_id() {
            return Err(ExecutionError::StateArenaMismatch {
                batch: batch.arena_id(),
                executor: self.program.state_schema().arena_id(),
            });
        }
        self.batch.lower_into(batch)?;
        let output = self.program.execute(&self.batch)?;
        let tokens = self.resolve_generation_output(&self.batch, output)?;
        self.completions
            .submit(ExecutionOutput::Generation { requests: tokens })
    }

    fn poll(
        &mut self,
        completion: CompletionId,
    ) -> Result<Option<ExecutionOutput>, ExecutionError> {
        self.completions.poll(completion)
    }

    fn take_execution_stats(&mut self) -> ExecutionStats {
        self.program.take_execution_stats()
    }
}

impl<P> CudaExecutor<P> {
    fn resolve_generation_output(
        &self,
        batch: &CudaBatch,
        output: ProgramOutput,
    ) -> Result<Vec<GeneratedTokens>, ExecutionError> {
        let sampled = match output {
            ProgramOutput::None if batch.samples.is_empty() => Vec::new(),
            ProgramOutput::None => {
                return Err(ExecutionError::MissingOutput {
                    samples: batch.samples.len(),
                });
            }
            ProgramOutput::Tokens(tokens) if tokens.len() == batch.samples.len() => tokens,
            ProgramOutput::Tokens(tokens) => {
                return Err(ExecutionError::TokenOutputCount {
                    expected: batch.samples.len(),
                    actual: tokens.len(),
                });
            }
            ProgramOutput::HostLogits { values, vocab_size } => {
                let valid = vocab_size > 0
                    && batch.samples.len().checked_mul(vocab_size) == Some(values.len());
                if !valid {
                    return Err(ExecutionError::LogitOutputShape {
                        samples: batch.samples.len(),
                        vocab_size,
                        actual: values.len(),
                    });
                }
                batch
                    .samples
                    .iter()
                    .zip(values.chunks_exact(vocab_size))
                    .map(|(sample, logits)| self.host_sampler.sample(logits, sample.sampling))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        Ok(batch
            .samples
            .iter()
            .zip(sampled)
            .map(|(sample, token_id)| GeneratedTokens::one(sample.request_id, token_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda::batch::SampleTarget,
        engine::{RequestId, SamplingInput},
    };

    fn sampled_batch(samples: usize) -> CudaBatch {
        let sampling = SamplingInput::try_new(0.0, 1.0, 0.0).unwrap();
        CudaBatch {
            token_ids: vec![0; samples],
            positions: vec![0; samples],
            current_slots: vec![0; samples],
            request_indices: (0..samples as u32).collect(),
            context_lengths: vec![1; samples],
            context_storage: vec![vec![0]; samples],
            num_requests: samples,
            sample_rows: (0..samples as u32).collect(),
            samples: (0..samples)
                .map(|_| SampleTarget {
                    request_id: RequestId::now_v7(),
                    sampling,
                })
                .collect(),
            all_samples_greedy: true,
            num_prefill_tokens: 0,
        }
    }

    #[test]
    fn rejects_program_output_cardinality_mismatches() {
        let executor = CudaExecutor::new(());
        let batch = sampled_batch(2);
        assert!(matches!(
            executor.resolve_generation_output(&batch, ProgramOutput::None),
            Err(ExecutionError::MissingOutput { samples: 2 })
        ));
        assert!(matches!(
            executor.resolve_generation_output(&batch, ProgramOutput::Tokens(Vec::new())),
            Err(ExecutionError::TokenOutputCount {
                expected: 2,
                actual: 0
            })
        ));
        assert!(matches!(
            executor.resolve_generation_output(
                &batch,
                ProgramOutput::HostLogits {
                    values: vec![0.0; 5],
                    vocab_size: 3,
                }
            ),
            Err(ExecutionError::LogitOutputShape {
                samples: 2,
                vocab_size: 3,
                actual: 5
            })
        ));
    }

    #[test]
    fn resolves_one_host_logit_row_per_sample() {
        let executor = CudaExecutor::new(());
        let batch = sampled_batch(2);
        let tokens = executor
            .resolve_generation_output(
                &batch,
                ProgramOutput::HostLogits {
                    values: vec![0.0, 2.0, 1.0, 4.0, 1.0, 0.0],
                    vocab_size: 3,
                },
            )
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].request_id(), batch.samples[0].request_id);
        assert_eq!(tokens[0].token_ids(), [TokenId::new(1)]);
        assert_eq!(tokens[1].request_id(), batch.samples[1].request_id);
        assert_eq!(tokens[1].token_ids(), [TokenId::new(0)]);
    }
}
