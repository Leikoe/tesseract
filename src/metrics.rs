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
        out
    }
}
