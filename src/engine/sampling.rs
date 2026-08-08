use thiserror::Error;

use super::{SamplingInput, TokenId};

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SamplingError {
    #[error("sampler received empty or non-finite logits")]
    InvalidLogits,
    #[error("sampler probability mass is invalid")]
    InvalidProbabilityMass,
    #[error("sampler failed to select a token")]
    NoSelection,
    #[error("sampled token index does not fit TokenId")]
    TokenIdOutOfRange,
}

/// Explicit host implementation for logits that are already resident on the
/// CPU. CUDA executors may use it for stochastic eager output, but must not
/// silently copy device logits here as a fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostLogitsSampler;

impl HostLogitsSampler {
    pub fn sample(self, logits: &[f32], sampling: SamplingInput) -> Result<TokenId, SamplingError> {
        if logits.is_empty() || logits.iter().any(|logit| !logit.is_finite()) {
            return Err(SamplingError::InvalidLogits);
        }
        if sampling.is_greedy() {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(token, _)| token)
                .ok_or(SamplingError::InvalidLogits)?;
            return token_id(token);
        }

        let inverse_temperature = 1.0 / sampling.temperature();
        let max = logits
            .iter()
            .copied()
            .map(|logit| logit * inverse_temperature)
            .max_by(f32::total_cmp)
            .ok_or(SamplingError::InvalidLogits)?;
        let mut candidates = logits
            .iter()
            .enumerate()
            .map(|(token, logit)| (token, ((*logit * inverse_temperature - max) as f64).exp()))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let total = candidates.iter().map(|(_, weight)| weight).sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(SamplingError::InvalidProbabilityMass);
        }

        let cutoff = total * sampling.top_p() as f64;
        let mut retained_mass = 0.0;
        let mut retained = 0usize;
        for (_, weight) in &candidates {
            retained_mass += weight;
            retained += 1;
            if retained_mass >= cutoff {
                break;
            }
        }
        let draw = sampling.random_sample() * retained_mass;
        let mut cumulative = 0.0;
        for (token, weight) in candidates.into_iter().take(retained) {
            cumulative += weight;
            if draw < cumulative {
                return token_id(token);
            }
        }
        Err(SamplingError::NoSelection)
    }
}

fn token_id(token: usize) -> Result<TokenId, SamplingError> {
    u32::try_from(token)
        .map(TokenId::new)
        .map_err(|_| SamplingError::TokenIdOutOfRange)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn greedy_uses_total_order_and_first_maximum() {
        let sampling = SamplingInput::try_new(0.0, 1.0, 0.0).unwrap();
        assert_eq!(
            HostLogitsSampler
                .sample(&[-1.0, 4.0, 4.0, 3.0], sampling)
                .unwrap()
                .get(),
            2
        );
    }

    #[test]
    fn rejects_invalid_logits() {
        let sampling = SamplingInput::try_new(0.8, 0.9, 0.5).unwrap();
        assert_eq!(
            HostLogitsSampler.sample(&[], sampling),
            Err(SamplingError::InvalidLogits)
        );
        assert_eq!(
            HostLogitsSampler.sample(&[0.0, f32::NAN], sampling),
            Err(SamplingError::InvalidLogits)
        );
    }

    proptest! {
        #[test]
        fn finite_logits_always_select_an_in_range_token(
            logits in prop::collection::vec(-100.0f32..100.0, 1..512),
            temperature in 0.01f32..4.0,
            top_p in 0.01f32..=1.0,
            draw in 0.0f64..1.0,
        ) {
            let sampling = SamplingInput::try_new(temperature, top_p, draw).unwrap();
            let token = HostLogitsSampler.sample(&logits, sampling).unwrap();
            prop_assert!((token.get() as usize) < logits.len());
        }
    }
}
