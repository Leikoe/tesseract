use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, ensure};
use tokenizers::Tokenizer;

use crate::config::{BenchmarkConfig, BenchmarkDataset};

const BUILTIN_PROMPTS: &[&str] = &[
    "What is the capital of France? Answer briefly.",
    "Count from one to ten, one number per line.",
    "In two sentences, explain why the sky appears blue.",
    "Write a short Rust function that adds two i32 values.",
    "Name three practical uses for a priority queue.",
    "Summarize the difference between TCP and UDP in one paragraph.",
    "Give four tips for debugging a production service.",
    "What is 17 multiplied by 23? Show the result only.",
];

pub(super) struct Sample {
    pub(super) prompt: String,
    pub(super) expected_input_tokens: Option<usize>,
    pub(super) output_tokens: usize,
}

pub(super) fn prepare(config: &BenchmarkConfig) -> Result<Vec<Sample>> {
    match config.dataset {
        BenchmarkDataset::Builtin => Ok(prepare_builtin(config)),
        BenchmarkDataset::Random => prepare_random(config),
    }
}

fn prepare_builtin(config: &BenchmarkConfig) -> Vec<Sample> {
    let prompts = match config.prompt.as_deref() {
        Some(prompt) => vec![prompt],
        None => BUILTIN_PROMPTS.to_vec(),
    };
    (0..config.num_prompts)
        .map(|index| Sample {
            prompt: prompts[index % prompts.len()].to_owned(),
            expected_input_tokens: None,
            output_tokens: config.output_len,
        })
        .collect()
}

fn prepare_random(config: &BenchmarkConfig) -> Result<Vec<Sample>> {
    ensure!(
        config.prompt.is_none(),
        "prompt is only valid with --dataset builtin"
    );
    let tokenizer_path = tokenizer_path(config)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?;
    let special_tokens = tokenizer
        .encode("", true)
        .map_err(|error| anyhow!(error.to_string()))?
        .len()
        .saturating_sub(
            tokenizer
                .encode("", false)
                .map_err(|error| anyhow!(error.to_string()))?
                .len(),
        );
    let input_base = config.input_len.saturating_sub(special_tokens);
    let (input_low, input_high) = length_bounds(input_base, config.length_variation);
    let (output_low, output_high) = length_bounds(config.output_len, config.length_variation);
    let output_low = output_low.max(1);
    let output_high = output_high.max(1);
    ensure!(
        config.shared_prefix_len + input_low > 0,
        "random workload can produce an empty prompt; increase input-len or shared-prefix-len"
    );

    let special_ids = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, token)| token.special.then_some(id))
        .collect::<std::collections::HashSet<_>>();
    let mut allowed_ids = tokenizer
        .get_vocab(true)
        .into_values()
        .filter(|id| !special_ids.contains(id))
        .collect::<Vec<_>>();
    allowed_ids.sort_unstable();
    allowed_ids.dedup();
    ensure!(
        !allowed_ids.is_empty(),
        "tokenizer has no non-special tokens"
    );

    let mut rng = Rng::new(config.seed);
    let prefix_ids = (0..config.shared_prefix_len)
        .map(|_| allowed_ids[rng.index(allowed_ids.len())])
        .collect::<Vec<_>>();
    let prefix_ids = stabilize(
        &tokenizer,
        prefix_ids,
        config.shared_prefix_len,
        &allowed_ids,
        &mut rng,
    )?
    .1;

    let mut samples = Vec::with_capacity(config.num_prompts);
    for index in 0..config.num_prompts {
        let input_len = rng.inclusive(input_low, input_high);
        let output_len = rng.inclusive(output_low, output_high);
        let offset = rng.index(allowed_ids.len());
        let inner = (0..input_len).map(|position| {
            let location = (offset + index + position) % allowed_ids.len();
            allowed_ids[location]
        });
        let token_ids = prefix_ids.iter().copied().chain(inner).collect();
        let target = config.shared_prefix_len + input_len;
        let (prompt, adjusted) = stabilize(&tokenizer, token_ids, target, &allowed_ids, &mut rng)
            .with_context(|| format!("construct random prompt {index}"))?;
        samples.push(Sample {
            prompt,
            expected_input_tokens: Some(adjusted.len() + special_tokens),
            output_tokens: output_len,
        });
    }
    Ok(samples)
}

fn tokenizer_path(config: &BenchmarkConfig) -> Result<PathBuf> {
    let path = config
        .tokenizer
        .clone()
        .context("random dataset requires --tokenizer PATH")?;
    Ok(if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path
    })
}

fn stabilize(
    tokenizer: &Tokenizer,
    mut token_ids: Vec<u32>,
    target: usize,
    allowed_ids: &[u32],
    rng: &mut Rng,
) -> Result<(String, Vec<u32>)> {
    for _ in 0..=10 {
        let prompt = tokenizer
            .decode(&token_ids, true)
            .map_err(|error| anyhow!(error.to_string()))?;
        let adjusted = tokenizer
            .encode(prompt.clone(), false)
            .map_err(|error| anyhow!(error.to_string()))?
            .get_ids()
            .to_vec();
        match adjusted.len().cmp(&target) {
            std::cmp::Ordering::Equal => return Ok((prompt, adjusted)),
            std::cmp::Ordering::Less => {
                token_ids = adjusted;
                token_ids.extend(
                    (0..target - token_ids.len())
                        .map(|_| allowed_ids[rng.index(allowed_ids.len())]),
                );
            }
            std::cmp::Ordering::Greater => {
                token_ids = adjusted[..target].to_vec();
            }
        }
    }
    let prompt = tokenizer
        .decode(&token_ids, true)
        .map_err(|error| anyhow!(error.to_string()))?;
    let adjusted = tokenizer
        .encode(prompt.clone(), false)
        .map_err(|error| anyhow!(error.to_string()))?
        .get_ids()
        .to_vec();
    ensure!(
        adjusted.len() == target,
        "decode and re-encode converged to {} tokens instead of {target}",
        adjusted.len()
    );
    Ok((prompt, adjusted))
}

fn length_bounds(base: usize, variation: f64) -> (usize, usize) {
    let base = base as f64;
    (
        (base * (1.0 - variation)).floor() as usize,
        (base * (1.0 + variation)).ceil() as usize,
    )
}

struct Rng {
    state: u64,
}

impl Rng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        self.inclusive(0, length - 1)
    }

    fn inclusive(&mut self, low: usize, high: usize) -> usize {
        let width = (high - low) as u64 + 1;
        let rejection = u64::MAX - u64::MAX % width;
        loop {
            let value = self.next();
            if value < rejection {
                return low + (value % width) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rng, length_bounds};

    #[test]
    fn zero_variation_is_an_exact_length() {
        assert_eq!(length_bounds(1024, 0.0), (1024, 1024));
    }

    #[test]
    fn variation_is_symmetric_and_outward_rounded() {
        assert_eq!(length_bounds(9, 0.25), (6, 12));
    }

    #[test]
    fn sampling_is_seeded_and_inclusive() {
        let mut first = Rng::new(42);
        let mut second = Rng::new(42);
        let a = (0..100).map(|_| first.inclusive(3, 7)).collect::<Vec<_>>();
        let b = (0..100).map(|_| second.inclusive(3, 7)).collect::<Vec<_>>();
        assert_eq!(a, b);
        assert!(a.iter().all(|value| (3..=7).contains(value)));
        assert!(a.contains(&3));
        assert!(a.contains(&7));
    }
}
