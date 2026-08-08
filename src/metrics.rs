use std::{
    fmt::Write,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
pub struct Metrics {
    ready: AtomicBool,
    queue_depth: AtomicUsize,
    running_requests: AtomicUsize,
    kv_tokens_used: AtomicUsize,
    requests_total: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    requests_cancelled: AtomicU64,
    prompt_tokens: AtomicU64,
    generated_tokens: AtomicU64,
    engine_steps: AtomicU64,
    engine_batches: AtomicU64,
    batch_items: AtomicU64,
    max_batch_size: AtomicUsize,
    ttft_microseconds: AtomicU64,
    ttft_count: AtomicU64,
    inter_token_microseconds: AtomicU64,
    inter_token_count: AtomicU64,
    request_microseconds: AtomicU64,
    request_duration_count: AtomicU64,
    eager_forwards: AtomicU64,
    graph_replays: AtomicU64,
    graph_captures: AtomicU64,
    packed_decode_forwards: AtomicU64,
    packed_decode_requests: AtomicU64,
}

impl Metrics {
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub(crate) fn set_queue_depth(&self, value: usize) {
        self.queue_depth.store(value, Ordering::Relaxed);
    }

    pub(crate) fn set_running_requests(&self, value: usize) {
        self.running_requests.store(value, Ordering::Relaxed);
    }

    pub(crate) fn set_kv_tokens_used(&self, value: usize) {
        self.kv_tokens_used.store(value, Ordering::Relaxed);
    }

    pub(crate) fn request_started(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_completed(&self) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_failed(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_cancelled(&self) {
        self.requests_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn add_prompt_tokens(&self, value: usize) {
        self.prompt_tokens
            .fetch_add(value as u64, Ordering::Relaxed);
    }

    pub(crate) fn token_generated(&self) {
        self.generated_tokens.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn engine_step(&self) {
        self.engine_steps.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn observe_batch(&self, size: usize) {
        self.engine_batches.fetch_add(1, Ordering::Relaxed);
        self.batch_items.fetch_add(size as u64, Ordering::Relaxed);
        self.max_batch_size.fetch_max(size, Ordering::Relaxed);
    }

    pub(crate) fn observe_ttft(&self, duration: std::time::Duration) {
        self.ttft_microseconds
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn observe_inter_token(&self, duration: std::time::Duration) {
        self.inter_token_microseconds
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        self.inter_token_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn observe_request_duration(&self, duration: std::time::Duration) {
        self.request_microseconds
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        self.request_duration_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn add_execution(&self, stats: crate::engine::ExecutionStats) {
        self.eager_forwards
            .fetch_add(stats.eager_forwards, Ordering::Relaxed);
        self.graph_replays
            .fetch_add(stats.graph_replays, Ordering::Relaxed);
        self.graph_captures
            .fetch_add(stats.graph_captures, Ordering::Relaxed);
        self.packed_decode_forwards
            .fetch_add(stats.packed_decode_forwards, Ordering::Relaxed);
        self.packed_decode_requests
            .fetch_add(stats.packed_decode_requests, Ordering::Relaxed);
    }

    pub fn prometheus(&self) -> String {
        let mut out = String::with_capacity(1024);
        macro_rules! metric {
            ($name:literal, $help:literal, $kind:literal, $value:expr) => {{
                writeln!(out, concat!("# HELP ", $name, " ", $help)).unwrap();
                writeln!(out, concat!("# TYPE ", $name, " ", $kind)).unwrap();
                writeln!(out, concat!($name, " {}"), $value).unwrap();
            }};
        }

        metric!(
            "tesseract_ready",
            "Whether the model worker is ready",
            "gauge",
            usize::from(self.is_ready())
        );
        metric!(
            "tesseract_queue_depth",
            "Requests waiting for execution",
            "gauge",
            self.queue_depth.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_running_requests",
            "Requests in the running batch",
            "gauge",
            self.running_requests.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_kv_tokens_used",
            "Allocated physical KV token slots",
            "gauge",
            self.kv_tokens_used.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_requests_total",
            "Admitted generation requests",
            "counter",
            self.requests_total.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_requests_completed_total",
            "Successfully completed generation requests",
            "counter",
            self.requests_completed.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_requests_failed_total",
            "Failed generation requests",
            "counter",
            self.requests_failed.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_requests_cancelled_total",
            "Cancelled generation requests",
            "counter",
            self.requests_cancelled.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_prompt_tokens_total",
            "Prompt tokens admitted",
            "counter",
            self.prompt_tokens.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_generated_tokens_total",
            "Tokens generated",
            "counter",
            self.generated_tokens.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_engine_steps_total",
            "Model execution steps",
            "counter",
            self.engine_steps.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_engine_batches_total",
            "Model-executor batches",
            "counter",
            self.engine_batches.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_batch_items_total",
            "Scheduled request items across executor batches",
            "counter",
            self.batch_items.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_max_batch_size",
            "Largest scheduled executor batch",
            "gauge",
            self.max_batch_size.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_eager_forwards_total",
            "Model forwards executed through the eager fallback",
            "counter",
            self.eager_forwards.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_cuda_graph_replays_total",
            "Model forwards executed by CUDA graph replay",
            "counter",
            self.graph_replays.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_cuda_graph_captures_total",
            "Full-model CUDA decode graphs captured",
            "counter",
            self.graph_captures.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_packed_decode_forwards_total",
            "Packed multi-request decode forwards executed",
            "counter",
            self.packed_decode_forwards.load(Ordering::Relaxed)
        );
        metric!(
            "tesseract_packed_decode_requests_total",
            "Request rows executed by packed multi-request decode",
            "counter",
            self.packed_decode_requests.load(Ordering::Relaxed)
        );
        timing_metrics(
            &mut out,
            "tesseract_time_to_first_token_seconds",
            "Time from admission to first generated token",
            &self.ttft_microseconds,
            &self.ttft_count,
        );
        timing_metrics(
            &mut out,
            "tesseract_inter_token_seconds",
            "Time between generated tokens",
            &self.inter_token_microseconds,
            &self.inter_token_count,
        );
        timing_metrics(
            &mut out,
            "tesseract_request_duration_seconds",
            "End-to-end request lifetime",
            &self.request_microseconds,
            &self.request_duration_count,
        );
        out
    }
}

fn timing_metrics(
    out: &mut String,
    name: &str,
    help: &str,
    microseconds: &AtomicU64,
    count: &AtomicU64,
) {
    writeln!(out, "# HELP {name} {help}").unwrap();
    writeln!(out, "# TYPE {name} summary").unwrap();
    writeln!(
        out,
        "{name}_sum {}",
        microseconds.load(Ordering::Relaxed) as f64 / 1_000_000.0
    )
    .unwrap();
    writeln!(out, "{name}_count {}", count.load(Ordering::Relaxed)).unwrap();
}
