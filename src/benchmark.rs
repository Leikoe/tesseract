use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, ensure};
use futures_util::StreamExt;
use reqwest::{
    Client, Response, Url,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::config::{BenchmarkApi, BenchmarkConfig};

mod dataset;

use dataset::prepare;

#[derive(Debug, Serialize)]
struct RequestResult {
    index: usize,
    prompt: String,
    generated_prompt_tokens: Option<usize>,
    requested_output_tokens: usize,
    latency_seconds: f64,
    ttft_seconds: f64,
    mean_inter_token_seconds: f64,
    #[serde(skip)]
    inter_token_intervals: Vec<f64>,
    token_events: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum RequestOutcome {
    Success {
        #[serde(flatten)]
        result: RequestResult,
    },
    Failed {
        index: usize,
        prompt: String,
        error: String,
    },
}

struct RequestSpec<'a> {
    endpoint: &'a Url,
    api: BenchmarkApi,
    model: &'a str,
    prompt: &'a str,
    generated_prompt_tokens: Option<usize>,
    output_len: usize,
    seed: u64,
    index: usize,
}

#[derive(Debug, Serialize)]
struct Distribution {
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    successful_requests: usize,
    failed_requests: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    wall_seconds: f64,
    request_throughput: f64,
    prompt_tokens_per_second: f64,
    completion_tokens_per_second: f64,
    total_tokens_per_second: f64,
    peak_concurrency: usize,
    request_latency_seconds: Distribution,
    ttft_seconds: Distribution,
    time_per_output_token_seconds: Distribution,
    inter_token_seconds: Distribution,
}

#[derive(Debug, Serialize)]
struct Report<'a> {
    schema_version: u32,
    timestamp_unix_seconds: u64,
    base_url: &'a str,
    api: &'a str,
    endpoint: &'a str,
    model: &'a str,
    dataset: &'a str,
    input_len: usize,
    length_variation: f64,
    shared_prefix_len: usize,
    num_prompts: usize,
    max_concurrency: usize,
    request_rate: Option<f64>,
    output_len: usize,
    warmup_requests: usize,
    seed: u64,
    metadata: &'a BTreeMap<String, String>,
    summary: &'a Summary,
    results: Option<&'a [RequestOutcome]>,
}

#[derive(Default)]
struct Concurrency {
    current: AtomicUsize,
    peak: AtomicUsize,
}

impl Concurrency {
    fn enter(&self) -> ConcurrencyGuard<'_> {
        let current = self.current.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak.fetch_max(current, Ordering::AcqRel);
        ConcurrencyGuard(self)
    }
}

struct ConcurrencyGuard<'a>(&'a Concurrency);

impl Drop for ConcurrencyGuard<'_> {
    fn drop(&mut self) {
        self.0.current.fetch_sub(1, Ordering::AcqRel);
    }
}

pub async fn run(config: BenchmarkConfig) -> Result<()> {
    validate(&config)?;
    let endpoint = endpoint_url(&config)?;
    let headers = headers(&config.header)?;
    let client = Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs_f64(config.timeout_seconds))
        .build()
        .context("build benchmark HTTP client")?;

    let samples = prepare(&config)?;
    let max_concurrency = config
        .max_concurrency
        .unwrap_or(config.num_prompts)
        .min(config.num_prompts);
    let warmup_permits = Arc::new(Semaphore::new(max_concurrency));
    let mut warmups = JoinSet::new();
    for index in 0..config.warmup_requests {
        let sample = &samples[index % samples.len()];
        let client = client.clone();
        let endpoint = endpoint.clone();
        let api = config.api;
        let model = config.model.clone();
        let prompt = sample.prompt.clone();
        let generated_prompt_tokens = sample.generated_prompt_tokens;
        let output_len = sample.output_tokens.min(32);
        let seed = config.seed.wrapping_add(index as u64);
        let permits = Arc::clone(&warmup_permits);
        warmups.spawn(async move {
            let _permit = permits
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("warmup concurrency limiter closed"))?;
            request_once(
                &client,
                RequestSpec {
                    endpoint: &endpoint,
                    api,
                    model: &model,
                    prompt: &prompt,
                    generated_prompt_tokens,
                    output_len,
                    seed,
                    index,
                },
            )
            .await
            .with_context(|| format!("warmup request {index}"))?;
            Ok::<_, anyhow::Error>(())
        });
    }
    while let Some(result) = warmups.join_next().await {
        result.context("warmup task panicked")??;
    }

    let arrivals = arrival_offsets(config.num_prompts, config.request_rate, config.seed);
    let started = Instant::now();
    let permits = Arc::new(Semaphore::new(max_concurrency));
    let concurrency = Arc::new(Concurrency::default());
    let mut tasks = JoinSet::new();
    for index in 0..config.num_prompts {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let api = config.api;
        let model = config.model.clone();
        let sample = &samples[index];
        let prompt = sample.prompt.clone();
        let generated_prompt_tokens = sample.generated_prompt_tokens;
        let permits = Arc::clone(&permits);
        let concurrency = Arc::clone(&concurrency);
        let due = started + arrivals[index];
        let output_len = sample.output_tokens;
        let seed = config.seed.wrapping_add(index as u64);
        tasks.spawn(async move {
            tokio::time::sleep_until(due.into()).await;
            let _permit = match permits.acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => {
                    return (
                        index,
                        RequestOutcome::Failed {
                            index,
                            prompt,
                            error: "benchmark concurrency limiter closed".to_owned(),
                        },
                    );
                }
            };
            let _in_flight = concurrency.enter();
            let outcome = match request_once(
                &client,
                RequestSpec {
                    endpoint: &endpoint,
                    api,
                    model: &model,
                    prompt: &prompt,
                    generated_prompt_tokens,
                    output_len,
                    seed,
                    index,
                },
            )
            .await
            .with_context(|| format!("request {index}"))
            {
                Ok(result) => RequestOutcome::Success { result },
                Err(error) => RequestOutcome::Failed {
                    index,
                    prompt,
                    error: format!("{error:#}"),
                },
            };
            (index, outcome)
        });
    }
    let mut indexed = Vec::with_capacity(config.num_prompts);
    while let Some(result) = tasks.join_next().await {
        indexed.push(result.context("benchmark task panicked")?);
    }
    let wall_seconds = started.elapsed().as_secs_f64();

    let mut slots: Vec<Option<RequestOutcome>> = (0..config.num_prompts).map(|_| None).collect();
    for (index, outcome) in indexed {
        slots[index] = Some(outcome);
    }
    let results = slots
        .into_iter()
        .enumerate()
        .map(|(index, result)| result.ok_or_else(|| anyhow!("missing result {index}")))
        .collect::<Result<Vec<_>>>()?;
    let summary = summarize(
        &results,
        wall_seconds,
        concurrency.peak.load(Ordering::Acquire),
    );

    println!("Successful requests: {}", summary.successful_requests);
    println!("Failed requests: {}", summary.failed_requests);
    println!("Benchmark duration: {:.2} s", summary.wall_seconds);
    println!(
        "Request throughput: {:.2} req/s",
        summary.request_throughput
    );
    println!(
        "Input token throughput: {:.2} tok/s",
        summary.prompt_tokens_per_second
    );
    println!(
        "Output token throughput: {:.2} tok/s",
        summary.completion_tokens_per_second
    );
    println!("Mean TTFT: {:.2} ms", summary.ttft_seconds.mean * 1000.0);
    println!("P99 TTFT: {:.2} ms", summary.ttft_seconds.p99 * 1000.0);
    println!(
        "Mean inter-token latency: {:.2} ms",
        summary.inter_token_seconds.mean * 1000.0
    );
    println!("Peak concurrency: {}", summary.peak_concurrency);

    if let Some(path) = &config.output {
        let metadata = metadata(&config.metadata)?;
        let report = Report {
            schema_version: 1,
            timestamp_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            base_url: &config.base_url,
            api: config.api.name(),
            endpoint: endpoint.as_str(),
            model: &config.model,
            dataset: config.dataset.name(),
            input_len: config.input_len,
            length_variation: config.length_variation,
            shared_prefix_len: config.shared_prefix_len,
            num_prompts: config.num_prompts,
            max_concurrency,
            request_rate: config
                .request_rate
                .is_finite()
                .then_some(config.request_rate),
            output_len: config.output_len,
            warmup_requests: config.warmup_requests,
            seed: config.seed,
            metadata: &metadata,
            summary: &summary,
            results: config.output_details.then_some(results.as_slice()),
        };
        write_report(path, &report)?;
        println!("Results written to {}", path.display());
    }
    Ok(())
}

fn validate(config: &BenchmarkConfig) -> Result<()> {
    ensure!(config.num_prompts > 0, "num-prompts must be positive");
    ensure!(config.output_len > 0, "output-len must be positive");
    ensure!(
        config.length_variation.is_finite() && (0.0..1.0).contains(&config.length_variation),
        "length-variation must be finite and in [0, 1)"
    );
    if let Some(concurrency) = config.max_concurrency {
        ensure!(concurrency > 0, "max-concurrency must be positive");
    }
    ensure!(
        config.request_rate.is_infinite() || config.request_rate > 0.0,
        "request-rate must be positive or inf"
    );
    ensure!(
        !config.request_rate.is_nan(),
        "request-rate must be positive or inf"
    );
    ensure!(
        config.timeout_seconds.is_finite() && config.timeout_seconds > 0.0,
        "timeout-seconds must be finite and positive"
    );
    endpoint_url(config)?;
    Ok(())
}

fn endpoint_url(config: &BenchmarkConfig) -> Result<Url> {
    let mut base = config.base_url.trim_end_matches('/').to_owned();
    base.push('/');
    let base = Url::parse(&base).context("parse base-url")?;
    let endpoint = config.endpoint.as_deref().unwrap_or(config.api.endpoint());
    base.join(endpoint.trim_start_matches('/'))
        .context("join endpoint to base-url")
}

fn headers(values: &[String]) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    for value in values {
        let (name, value) = value
            .split_once(':')
            .with_context(|| format!("header must have NAME:VALUE form: {value}"))?;
        let name = HeaderName::from_bytes(name.trim().as_bytes())
            .with_context(|| format!("invalid header name: {name}"))?;
        let value = HeaderValue::from_str(value.trim()).context("invalid header value")?;
        headers.append(name, value);
    }
    Ok(headers)
}

fn metadata(values: &[String]) -> Result<BTreeMap<String, String>> {
    values
        .iter()
        .map(|value| {
            let (key, value) = value
                .split_once('=')
                .with_context(|| format!("metadata must have KEY=VALUE form: {value}"))?;
            ensure!(!key.trim().is_empty(), "metadata key cannot be empty");
            Ok((key.trim().to_owned(), value.to_owned()))
        })
        .collect()
}

async fn request_once(client: &Client, request: RequestSpec<'_>) -> Result<RequestResult> {
    let started = Instant::now();
    let body = match request.api {
        BenchmarkApi::ChatCompletions => json!({
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.output_len,
            "temperature": 0.0,
            "seed": request.seed,
            "stream": true,
            "stream_options": {"include_usage": true},
        }),
        BenchmarkApi::Completions => json!({
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.output_len,
            "temperature": 0.0,
            "seed": request.seed,
            "stream": true,
            "stream_options": {"include_usage": true},
        }),
    };
    let response = client
        .post(request.endpoint.clone())
        .json(&body)
        .send()
        .await
        .context("send chat completion")?;
    let response = successful(response).await?;
    let mut stream = response.bytes_stream();
    let mut token_times = Vec::new();
    let mut usage = Value::Null;
    let mut finish_reason = None;
    let mut pending = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read SSE response")?;
        pending.push_str(&String::from_utf8_lossy(&chunk));
        consume_events(
            &mut pending,
            request.api,
            started,
            &mut token_times,
            &mut usage,
            &mut finish_reason,
        )?;
    }
    let latency_seconds = started.elapsed().as_secs_f64();
    let intervals = token_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let prompt_tokens = usage["prompt_tokens"]
        .as_u64()
        .context("stream ended without prompt token usage")? as usize;
    let completion_tokens = usage["completion_tokens"]
        .as_u64()
        .context("stream ended without completion token usage")?
        as usize;
    Ok(RequestResult {
        index: request.index,
        prompt: request.prompt.to_owned(),
        generated_prompt_tokens: request.generated_prompt_tokens,
        requested_output_tokens: request.output_len,
        latency_seconds,
        ttft_seconds: token_times.first().copied().unwrap_or(latency_seconds),
        mean_inter_token_seconds: mean(&intervals),
        inter_token_intervals: intervals,
        token_events: token_times.len(),
        prompt_tokens,
        completion_tokens,
        finish_reason,
    })
}

fn consume_events(
    pending: &mut String,
    api: BenchmarkApi,
    started: Instant,
    token_times: &mut Vec<f64>,
    usage: &mut Value,
    finish_reason: &mut Option<String>,
) -> Result<()> {
    while let Some((end, delimiter_len)) = event_boundary(pending) {
        let event = pending[..end].to_owned();
        pending.drain(..end + delimiter_len);
        for line in event.lines() {
            let Some(data) = line.trim_end_matches('\r').strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let event: Value = serde_json::from_str(data).context("parse SSE event")?;
            if !event["usage"].is_null() {
                *usage = event["usage"].clone();
            }
            let Some(choice) = event["choices"]
                .as_array()
                .and_then(|choices| choices.first())
            else {
                continue;
            };
            let text = match api {
                BenchmarkApi::ChatCompletions => choice["delta"]["content"].as_str(),
                BenchmarkApi::Completions => choice["text"].as_str(),
            };
            if text.is_some_and(|text| !text.is_empty()) {
                token_times.push(started.elapsed().as_secs_f64());
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                *finish_reason = Some(reason.to_owned());
            }
        }
    }
    Ok(())
}

fn event_boundary(pending: &str) -> Option<(usize, usize)> {
    let line_feed = pending.find("\n\n").map(|index| (index, 2));
    let carriage_return = pending.find("\r\n\r\n").map(|index| (index, 4));
    match (line_feed, carriage_return) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(boundary), None) | (None, Some(boundary)) => Some(boundary),
        (None, None) => None,
    }
}

async fn successful(response: Response) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    Err(anyhow!("chat completion returned HTTP {status}: {body}"))
}

fn arrival_offsets(count: usize, rate: f64, seed: u64) -> Vec<Duration> {
    if rate.is_infinite() {
        return vec![Duration::ZERO; count];
    }
    let mut state = seed;
    let mut elapsed = 0.0;
    let mut offsets = Vec::with_capacity(count);
    for index in 0..count {
        if index > 0 {
            let random = splitmix64(&mut state);
            let unit = ((random >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64);
            elapsed += -unit.ln() / rate;
        }
        offsets.push(Duration::from_secs_f64(elapsed));
    }
    offsets
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn summarize(results: &[RequestOutcome], wall_seconds: f64, peak_concurrency: usize) -> Summary {
    let successful = results
        .iter()
        .filter_map(|outcome| match outcome {
            RequestOutcome::Success { result } => Some(result),
            RequestOutcome::Failed { .. } => None,
        })
        .collect::<Vec<_>>();
    let prompt_tokens = successful.iter().map(|r| r.prompt_tokens).sum();
    let completion_tokens = successful.iter().map(|r| r.completion_tokens).sum();
    let latencies = successful
        .iter()
        .map(|r| r.latency_seconds)
        .collect::<Vec<_>>();
    let ttfts = successful
        .iter()
        .map(|r| r.ttft_seconds)
        .collect::<Vec<_>>();
    let tpots = successful
        .iter()
        .filter(|r| r.token_events > 1)
        .map(|r| r.mean_inter_token_seconds)
        .collect::<Vec<_>>();
    let inter_tokens = successful
        .iter()
        .flat_map(|r| r.inter_token_intervals.iter().copied())
        .collect::<Vec<_>>();
    Summary {
        successful_requests: successful.len(),
        failed_requests: results.len() - successful.len(),
        prompt_tokens,
        completion_tokens,
        wall_seconds,
        request_throughput: successful.len() as f64 / wall_seconds,
        prompt_tokens_per_second: prompt_tokens as f64 / wall_seconds,
        completion_tokens_per_second: completion_tokens as f64 / wall_seconds,
        total_tokens_per_second: (prompt_tokens + completion_tokens) as f64 / wall_seconds,
        peak_concurrency,
        request_latency_seconds: distribution(&latencies),
        ttft_seconds: distribution(&ttfts),
        time_per_output_token_seconds: distribution(&tpots),
        inter_token_seconds: distribution(&inter_tokens),
    }
}

fn distribution(values: &[f64]) -> Distribution {
    Distribution {
        mean: mean(values),
        p50: percentile(values, 0.50),
        p90: percentile(values, 0.90),
        p99: percentile(values, 0.99),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let position = (ordered.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(ordered.len() - 1);
    ordered[lower] * (1.0 - position.fract()) + ordered[upper] * position.fract()
}

fn write_report(path: &Path, report: &Report<'_>) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create result directory {}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    fs::write(path, bytes).with_context(|| format!("write benchmark results {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{arrival_offsets, consume_events, percentile};
    use crate::config::BenchmarkApi;
    use std::time::Instant;

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert_eq!(percentile(&[], 0.99), 0.0);
    }

    #[test]
    fn infinite_rate_releases_every_request_at_zero() {
        assert!(
            arrival_offsets(4, f64::INFINITY, 0)
                .iter()
                .all(|offset| offset.is_zero())
        );
    }

    #[test]
    fn poisson_arrivals_are_seeded_and_monotonic() {
        let first = arrival_offsets(16, 5.0, 42);
        let second = arrival_offsets(16, 5.0, 42);
        assert_eq!(first, second);
        assert!(first.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(first.last().unwrap() > &std::time::Duration::ZERO);
    }

    #[test]
    fn sse_parser_preserves_events_across_transport_chunks() {
        let mut pending = "data: {\"choices\":[{\"delta\":{\"content\":\"x".to_owned();
        let mut times = Vec::new();
        let mut usage = serde_json::Value::Null;
        let mut reason = None;
        consume_events(
            &mut pending,
            BenchmarkApi::ChatCompletions,
            Instant::now(),
            &mut times,
            &mut usage,
            &mut reason,
        )
        .unwrap();
        assert!(times.is_empty());

        pending.push_str(concat!(
            "\"},\"finish_reason\":null}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n"
        ));
        consume_events(
            &mut pending,
            BenchmarkApi::ChatCompletions,
            Instant::now(),
            &mut times,
            &mut usage,
            &mut reason,
        )
        .unwrap();
        assert_eq!(times.len(), 1);
        assert_eq!(usage["prompt_tokens"], 3);
        assert_eq!(usage["completion_tokens"], 1);
        assert_eq!(reason.as_deref(), Some("length"));
    }
}
