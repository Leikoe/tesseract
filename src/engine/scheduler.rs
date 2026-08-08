use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    fmt::Display,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use crate::model::Model;
use crate::{config::EngineConfig, metrics::Metrics};

use super::{
    Backend, FinishReason, GenerateRequest, GenerationEvent, RequestId, ScheduledWork, StepOutput,
    Usage, kv::KvSlots,
};

#[derive(Debug, Error)]
pub enum EngineSpawnError {
    #[error("failed to spawn engine thread: {0}")]
    Thread(#[from] std::io::Error),
    #[error("failed to initialize inference backend: {0}")]
    BackendInitialization(String),
    #[error("engine thread exited during initialization")]
    InitializationChannelClosed,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubmitError {
    #[error("the inference queue is full")]
    Overloaded,
    #[error("the inference engine is unavailable")]
    Unavailable,
}

enum Command {
    Start {
        request: GenerateRequest,
        output: mpsc::Sender<GenerationEvent>,
    },
    Cancel(RequestId),
    Shutdown(oneshot::Sender<()>),
}

#[derive(Clone)]
pub struct EngineHandle {
    commands: mpsc::Sender<Command>,
    metrics: Arc<Metrics>,
    model: Arc<dyn Model>,
    model_id: Arc<str>,
    output_buffer: usize,
    admission: Arc<Admission>,
}

struct Admission {
    active: AtomicUsize,
    limit: usize,
}

impl Admission {
    fn try_acquire(&self) -> bool {
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < self.limit).then_some(active + 1)
            })
            .is_ok()
    }

    fn release(&self) {
        let previous = self.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "admission permit released more than once");
    }
}

impl EngineHandle {
    pub fn spawn<B: Backend + Send>(
        backend: B,
        config: EngineConfig,
        command_capacity: usize,
        metrics: Arc<Metrics>,
    ) -> Result<Self, EngineSpawnError> {
        Self::spawn_with_factory(
            move || Ok::<B, Infallible>(backend),
            config,
            command_capacity,
            metrics,
        )
    }

    /// Builds the backend on the dedicated engine thread, then returns only
    /// after model loading and backend initialization have succeeded.
    pub fn spawn_with_factory<B, F, E>(
        factory: F,
        config: EngineConfig,
        command_capacity: usize,
        metrics: Arc<Metrics>,
    ) -> Result<Self, EngineSpawnError>
    where
        B: Backend,
        F: FnOnce() -> Result<B, E> + Send + 'static,
        E: Display,
    {
        let output_buffer = config.output_buffer;
        let admission_limit = config.max_pending + config.max_running;
        let (commands_tx, commands_rx) = mpsc::channel(command_capacity);
        let worker_metrics = Arc::clone(&metrics);
        let (initialized_tx, initialized_rx) = std::sync::mpsc::sync_channel(1);
        thread::Builder::new()
            .name("tesseract-engine".into())
            .spawn(move || match factory() {
                Ok(backend) => {
                    let model = backend.model();
                    if initialized_tx.send(Ok(model)).is_ok() {
                        EngineWorker::new(backend, config, worker_metrics).run(commands_rx);
                    }
                }
                Err(error) => {
                    let _ = initialized_tx.send(Err(error.to_string()));
                }
            })?;
        let model = initialized_rx
            .recv()
            .map_err(|_| EngineSpawnError::InitializationChannelClosed)?
            .map_err(EngineSpawnError::BackendInitialization)?;
        let model_id: Arc<str> = Arc::from(model.id());

        Ok(Self {
            commands: commands_tx,
            metrics,
            model,
            model_id,
            output_buffer,
            admission: Arc::new(Admission {
                active: AtomicUsize::new(0),
                limit: admission_limit,
            }),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }

    pub fn try_generate(&self, request: GenerateRequest) -> Result<RequestStream, SubmitError> {
        if !self.admission.try_acquire() {
            return Err(SubmitError::Overloaded);
        }
        let request_id = request.id;
        let (output_tx, output_rx) = mpsc::channel(self.output_buffer);
        if let Err(error) = self.commands.try_send(Command::Start {
            request,
            output: output_tx,
        }) {
            self.admission.release();
            return Err(match error {
                mpsc::error::TrySendError::Full(_) => SubmitError::Overloaded,
                mpsc::error::TrySendError::Closed(_) => SubmitError::Unavailable,
            });
        }

        Ok(RequestStream {
            request_id,
            output: output_rx,
            commands: self.commands.clone(),
            terminal: false,
            admission: Arc::clone(&self.admission),
            permit_released: false,
        })
    }

    pub fn try_cancel(&self, request_id: RequestId) -> Result<(), SubmitError> {
        self.commands
            .try_send(Command::Cancel(request_id))
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => SubmitError::Overloaded,
                mpsc::error::TrySendError::Closed(_) => SubmitError::Unavailable,
            })
    }

    pub async fn shutdown(&self, grace: Duration) -> Result<(), SubmitError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.commands
            .send(Command::Shutdown(ack_tx))
            .await
            .map_err(|_| SubmitError::Unavailable)?;
        tokio::time::timeout(grace, ack_rx)
            .await
            .map_err(|_| SubmitError::Unavailable)?
            .map_err(|_| SubmitError::Unavailable)
    }
}

pub struct RequestStream {
    request_id: RequestId,
    output: mpsc::Receiver<GenerationEvent>,
    commands: mpsc::Sender<Command>,
    terminal: bool,
    admission: Arc<Admission>,
    permit_released: bool,
}

impl RequestStream {
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        let event = self.output.recv().await;
        if matches!(
            event,
            Some(GenerationEvent::Finished { .. } | GenerationEvent::Failed { .. }) | None
        ) {
            self.terminal = true;
            self.release_permit();
        }
        event
    }

    fn release_permit(&mut self) {
        if !self.permit_released {
            self.admission.release();
            self.permit_released = true;
        }
    }
}

impl Drop for RequestStream {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = self.commands.try_send(Command::Cancel(self.request_id));
        }
        self.release_permit();
    }
}

struct RequestState {
    request: GenerateRequest,
    output: mpsc::Sender<GenerationEvent>,
    prompt_tokens: usize,
    computed_tokens: usize,
    generated_tokens: usize,
    pending_text: String,
    started_at: Instant,
    last_token_at: Option<Instant>,
}

impl RequestState {
    fn target_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    fn reservation_tokens(&self) -> usize {
        self.prompt_tokens + self.request.params.max_tokens
    }

    fn consume_text(&mut self, text: &str) -> (String, bool) {
        self.pending_text.push_str(text);
        if self.request.params.stop.is_empty() {
            return (std::mem::take(&mut self.pending_text), false);
        }

        let stop_at = self
            .request
            .params
            .stop
            .iter()
            .filter_map(|stop| self.pending_text.find(stop))
            .min();
        if let Some(index) = stop_at {
            let safe = self.pending_text[..index].to_owned();
            self.pending_text.clear();
            return (safe, true);
        }

        let keep = (1..=self.pending_text.len())
            .rev()
            .find(|&suffix_len| {
                self.pending_text
                    .is_char_boundary(self.pending_text.len() - suffix_len)
                    && self.request.params.stop.iter().any(|stop| {
                        stop.starts_with(&self.pending_text[self.pending_text.len() - suffix_len..])
                    })
            })
            .unwrap_or(0);
        let emit_len = self.pending_text.len() - keep;
        let emit = self.pending_text[..emit_len].to_owned();
        self.pending_text.drain(..emit_len);
        (emit, false)
    }
}

struct EngineWorker<B> {
    backend: B,
    config: EngineConfig,
    metrics: Arc<Metrics>,
    kv: KvSlots,
    requests: HashMap<RequestId, RequestState>,
    waiting: VecDeque<RequestId>,
    running: VecDeque<RequestId>,
    shutting_down: bool,
    shutdown_ack: Option<oneshot::Sender<()>>,
}

impl<B: Backend> EngineWorker<B> {
    fn new(backend: B, config: EngineConfig, metrics: Arc<Metrics>) -> Self {
        Self {
            backend,
            kv: KvSlots::new(config.kv_capacity_tokens),
            config,
            metrics,
            requests: HashMap::new(),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            shutting_down: false,
            shutdown_ack: None,
        }
    }

    fn run(mut self, mut commands: mpsc::Receiver<Command>) {
        self.metrics
            .add_backend_execution(self.backend.take_execution_stats());
        self.metrics.set_ready(true);
        loop {
            self.receive_commands(&mut commands);
            self.prune_disconnected();

            if self.shutting_down {
                self.cancel_all();
                break;
            }

            self.admit_waiting();
            let batch = self.build_batch();
            if batch.is_empty() {
                match commands.blocking_recv() {
                    Some(command) => self.handle_command(command),
                    None => {
                        self.shutting_down = true;
                    }
                }
                continue;
            }

            self.metrics.engine_step();
            self.metrics.observe_batch(batch.len());
            let result = self.backend.step(&batch);
            self.metrics
                .add_backend_execution(self.backend.take_execution_stats());
            match result {
                Ok(outputs) => self.apply_step(&batch, outputs),
                Err(error) => {
                    for work in batch {
                        self.fail_request(work.request_id, error.to_string());
                    }
                }
            }
            self.update_gauges();
        }

        self.metrics.set_ready(false);
        if let Err(error) = self.backend.shutdown() {
            tracing::error!(%error, "backend shutdown failed");
        }
        if let Some(ack) = self.shutdown_ack.take() {
            let _ = ack.send(());
        }
    }

    fn receive_commands(&mut self, commands: &mut mpsc::Receiver<Command>) {
        for _ in 0..64 {
            match commands.try_recv() {
                Ok(command) => self.handle_command(command),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.shutting_down = true;
                    break;
                }
            }
        }
    }

    fn handle_command(&mut self, command: Command) {
        match command {
            Command::Start { request, output } => self.add_request(request, output),
            Command::Cancel(request_id) => self.cancel_request(request_id),
            Command::Shutdown(ack) => {
                self.shutting_down = true;
                self.shutdown_ack = Some(ack);
            }
        }
    }

    fn add_request(&mut self, request: GenerateRequest, output: mpsc::Sender<GenerationEvent>) {
        if self.shutting_down
            || self.requests.len() >= self.config.max_pending + self.config.max_running
        {
            let _ = output.try_send(GenerationEvent::Failed {
                message: "the inference engine is overloaded".into(),
            });
            return;
        }
        if let Err(message) = request.params.validate() {
            let _ = output.try_send(GenerationEvent::Failed { message });
            return;
        }

        let id = request.id;
        let prepared = match self.backend.add_request(&request) {
            Ok(prepared) => prepared,
            Err(error) => {
                let _ = output.try_send(GenerationEvent::Failed {
                    message: error.to_string(),
                });
                self.metrics.request_failed();
                return;
            }
        };
        if prepared.prompt_tokens == 0 {
            self.backend.remove_request(id);
            let _ = output.try_send(GenerationEvent::Failed {
                message: "the rendered prompt is empty".into(),
            });
            self.metrics.request_failed();
            return;
        }
        let total = prepared.prompt_tokens + request.params.max_tokens;
        if total > self.config.max_sequence_length || total > self.kv.capacity() {
            self.backend.remove_request(id);
            let _ = output.try_send(GenerationEvent::Failed {
                message: format!(
                    "prompt plus max_tokens ({total}) exceeds the configured capacity"
                ),
            });
            self.metrics.request_failed();
            return;
        }

        self.metrics.request_started();
        self.metrics.add_prompt_tokens(prepared.prompt_tokens);
        self.requests.insert(
            id,
            RequestState {
                request,
                output,
                prompt_tokens: prepared.prompt_tokens,
                computed_tokens: 0,
                generated_tokens: 0,
                pending_text: String::new(),
                started_at: Instant::now(),
                last_token_at: None,
            },
        );
        self.waiting.push_back(id);
        self.update_gauges();
    }

    fn admit_waiting(&mut self) {
        let candidates = self.waiting.len();
        for _ in 0..candidates {
            if self.running.len() >= self.config.max_running {
                break;
            }
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let Some(state) = self.requests.get(&id) else {
                continue;
            };
            if self.kv.reserve(id, state.reservation_tokens()) {
                self.running.push_back(id);
            } else {
                self.waiting.push_back(id);
            }
        }
        self.update_gauges();
    }

    fn build_batch(&mut self) -> Vec<ScheduledWork> {
        let mut budget = self.config.max_batch_tokens;
        let ids: Vec<_> = self.running.iter().copied().collect();
        let mut batch = Vec::with_capacity(ids.len());

        for decode_phase in [true, false] {
            for id in &ids {
                if budget == 0 {
                    break;
                }
                let Some(state) = self.requests.get(id) else {
                    continue;
                };
                let is_decode = state.computed_tokens >= state.prompt_tokens;
                if is_decode != decode_phase {
                    continue;
                }
                let remaining = state.target_tokens().saturating_sub(state.computed_tokens);
                if remaining == 0 {
                    continue;
                }
                let limit = if is_decode {
                    1
                } else {
                    self.config.prefill_chunk_tokens
                };
                let num_tokens = remaining.min(limit).min(budget);
                let Some(kv_slots) = self.kv.allocate(*id, num_tokens) else {
                    continue;
                };
                batch.push(ScheduledWork {
                    request_id: *id,
                    position: state.computed_tokens,
                    num_tokens,
                    kv_slots,
                    sample: state.computed_tokens + num_tokens == state.target_tokens(),
                });
                budget -= num_tokens;
            }
        }
        batch
    }

    fn apply_step(&mut self, batch: &[ScheduledWork], outputs: Vec<StepOutput>) {
        let mut outputs: HashMap<_, _> = outputs
            .into_iter()
            .map(|output| (output.request_id, output))
            .collect();
        let mut finish = Vec::new();
        let mut failed = Vec::new();

        for work in batch {
            let Some(state) = self.requests.get_mut(&work.request_id) else {
                continue;
            };
            state.computed_tokens += work.num_tokens;
            if !work.sample {
                continue;
            }
            let Some(output) = outputs.remove(&work.request_id) else {
                failed.push((
                    work.request_id,
                    "backend omitted output for a sampled request".to_string(),
                ));
                continue;
            };

            state.generated_tokens += 1;
            self.metrics.token_generated();
            let now = Instant::now();
            match state.last_token_at {
                Some(previous) => self.metrics.observe_inter_token(now - previous),
                None => {
                    self.metrics.observe_ttft(now - state.started_at);
                }
            }
            state.last_token_at = Some(now);
            let (text, hit_stop) = state.consume_text(&output.text);
            if !text.is_empty()
                && state
                    .output
                    .try_send(GenerationEvent::Delta {
                        text,
                        token_id: output.token_id,
                    })
                    .is_err()
            {
                finish.push((work.request_id, FinishReason::Cancelled));
                continue;
            }

            let reason = if hit_stop || output.is_eos {
                Some(FinishReason::Stop)
            } else if state.generated_tokens >= state.request.params.max_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };
            if let Some(reason) = reason {
                finish.push((work.request_id, reason));
            }
        }

        for (id, message) in failed {
            self.fail_request(id, message);
        }
        for (id, reason) in finish {
            self.finish_request(id, reason);
        }
    }

    fn finish_request(&mut self, id: RequestId, reason: FinishReason) {
        let Some(mut state) = self.requests.remove(&id) else {
            return;
        };
        self.waiting.retain(|candidate| *candidate != id);
        self.running.retain(|candidate| *candidate != id);
        self.kv.release(id);
        self.backend.remove_request(id);
        self.metrics
            .observe_request_duration(state.started_at.elapsed());

        if !state.pending_text.is_empty() && reason != FinishReason::Cancelled {
            let _ = state.output.try_send(GenerationEvent::Delta {
                text: std::mem::take(&mut state.pending_text),
                token_id: None,
            });
        }
        let usage = Usage::new(state.prompt_tokens, state.generated_tokens);
        let _ = state
            .output
            .try_send(GenerationEvent::Finished { reason, usage });
        match reason {
            FinishReason::Cancelled => self.metrics.request_cancelled(),
            FinishReason::Error => self.metrics.request_failed(),
            FinishReason::Stop | FinishReason::Length => self.metrics.request_completed(),
        }
        self.update_gauges();
    }

    fn fail_request(&mut self, id: RequestId, message: String) {
        let Some(state) = self.requests.remove(&id) else {
            return;
        };
        self.waiting.retain(|candidate| *candidate != id);
        self.running.retain(|candidate| *candidate != id);
        self.kv.release(id);
        self.backend.remove_request(id);
        let _ = state.output.try_send(GenerationEvent::Failed { message });
        self.metrics.request_failed();
        self.update_gauges();
    }

    fn cancel_request(&mut self, id: RequestId) {
        self.finish_request(id, FinishReason::Cancelled);
    }

    fn prune_disconnected(&mut self) {
        let disconnected: Vec<_> = self
            .requests
            .iter()
            .filter_map(|(id, state)| state.output.is_closed().then_some(*id))
            .collect();
        for id in disconnected {
            self.cancel_request(id);
        }
    }

    fn cancel_all(&mut self) {
        let ids: Vec<_> = self.requests.keys().copied().collect();
        for id in ids {
            self.cancel_request(id);
        }
    }

    fn update_gauges(&self) {
        self.metrics.set_queue_depth(self.waiting.len());
        self.metrics.set_running_requests(self.running.len());
        self.metrics.set_kv_tokens_used(self.kv.used());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{GenerationParams, testing::DeterministicBackend};
    use std::collections::HashSet;

    fn config() -> EngineConfig {
        EngineConfig {
            max_pending: 8,
            max_running: 2,
            max_batch_tokens: 8,
            prefill_chunk_tokens: 2,
            max_sequence_length: 64,
            kv_capacity_tokens: 64,
            output_buffer: 8,
        }
    }

    fn request(max_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            id: RequestId::now_v7(),
            prompt: "one two three four".into(),
            params: GenerationParams {
                max_tokens,
                temperature: 0.0,
                top_p: 1.0,
                seed: 7,
                stop: vec![],
            },
        }
    }

    #[tokio::test]
    async fn produces_deltas_and_length_finish() {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            DeterministicBackend::new("test-model"),
            config(),
            8,
            metrics,
        )
        .unwrap();
        let mut stream = engine.try_generate(request(3)).unwrap();
        let mut text = String::new();
        let mut usage = None;
        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Delta { text: delta, .. } => text.push_str(&delta),
                GenerationEvent::Finished {
                    reason,
                    usage: event_usage,
                } => {
                    assert_eq!(reason, FinishReason::Length);
                    usage = Some(event_usage);
                    break;
                }
                GenerationEvent::Failed { message } => panic!("generation failed: {message}"),
            }
        }
        assert_eq!(text, " token0 token1 token2");
        assert_eq!(usage, Some(Usage::new(4, 3)));
    }

    #[tokio::test]
    async fn dropping_stream_cancels_and_releases_kv() {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            DeterministicBackend::new("test-model"),
            config(),
            8,
            Arc::clone(&metrics),
        )
        .unwrap();
        let stream = engine.try_generate(request(20)).unwrap();
        drop(stream);

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let rendered = metrics.prometheus();
                if rendered.contains("tesseract_kv_tokens_used 0")
                    && rendered.contains("tesseract_requests_cancelled_total 1")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn stop_string_is_not_emitted() {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            DeterministicBackend::new("test-model"),
            config(),
            8,
            metrics,
        )
        .unwrap();
        let mut request = request(5);
        request.params.stop = vec![" token1".into()];
        let mut stream = engine.try_generate(request).unwrap();
        let mut text = String::new();
        let mut reason = None;
        while let Some(event) = stream.recv().await {
            match event {
                GenerationEvent::Delta { text: delta, .. } => text.push_str(&delta),
                GenerationEvent::Finished {
                    reason: event_reason,
                    ..
                } => {
                    reason = Some(event_reason);
                    break;
                }
                GenerationEvent::Failed { message } => panic!("generation failed: {message}"),
            }
        }
        assert_eq!(text, " token0");
        assert_eq!(reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn frontend_admission_is_strictly_bounded() {
        let metrics = Arc::new(Metrics::default());
        let mut bounded = config();
        bounded.max_pending = 0;
        bounded.max_running = 1;
        let engine =
            EngineHandle::spawn(DeterministicBackend::new("test-model"), bounded, 8, metrics)
                .unwrap();
        let first = engine.try_generate(request(20)).unwrap();
        assert!(matches!(
            engine.try_generate(request(1)),
            Err(SubmitError::Overloaded)
        ));
        drop(first);
        assert!(engine.try_generate(request(1)).is_ok());
    }

    #[tokio::test]
    async fn explicit_cancellation_finishes_the_request() {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            DeterministicBackend::new("test-model"),
            config(),
            8,
            metrics,
        )
        .unwrap();
        let mut stream = engine.try_generate(request(20)).unwrap();
        engine.try_cancel(stream.request_id()).unwrap();
        while let Some(event) = stream.recv().await {
            if let GenerationEvent::Finished { reason, .. } = event {
                assert_eq!(reason, FinishReason::Cancelled);
                return;
            }
        }
        panic!("cancelled request closed without a terminal event");
    }

    #[tokio::test]
    async fn shutdown_cancels_active_requests_and_clears_readiness() {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            DeterministicBackend::new("test-model").with_step_delay(Duration::from_millis(25)),
            config(),
            8,
            Arc::clone(&metrics),
        )
        .unwrap();
        let mut stream = engine.try_generate(request(20)).unwrap();
        engine.shutdown(Duration::from_secs(1)).await.unwrap();
        assert!(!metrics.is_ready());
        while let Some(event) = stream.recv().await {
            if let GenerationEvent::Finished { reason, .. } = event {
                assert_eq!(reason, FinishReason::Cancelled);
                return;
            }
        }
        panic!("shutdown request closed without a terminal cancellation");
    }

    #[test]
    fn batch_builder_preserves_budget_chunk_and_slot_invariants() {
        for budget in 3..=9 {
            for chunk in 1..=4 {
                let mut test_config = config();
                test_config.max_running = 3;
                test_config.max_batch_tokens = budget;
                test_config.prefill_chunk_tokens = chunk;
                test_config.kv_capacity_tokens = 256;
                let metrics = Arc::new(Metrics::default());
                let backend = DeterministicBackend::new("test-model");
                let mut worker = EngineWorker::new(backend, test_config, metrics);
                let mut receivers = Vec::new();
                for prompt_tokens in [2usize, 5, 11] {
                    let (output, receiver) = mpsc::channel(8);
                    receivers.push(receiver);
                    let mut generated = request(3);
                    generated.prompt = vec!["token"; prompt_tokens].join(" ");
                    worker.add_request(generated, output);
                }
                worker.admit_waiting();
                let batch = worker.build_batch();
                assert!(batch.iter().map(|work| work.num_tokens).sum::<usize>() <= budget);
                assert!(batch.iter().all(|work| work.num_tokens <= chunk));
                let slots: Vec<_> = batch
                    .iter()
                    .flat_map(|work| work.kv_slots.iter().copied())
                    .collect();
                assert_eq!(
                    slots.iter().copied().collect::<HashSet<_>>().len(),
                    slots.len()
                );
                assert!(
                    batch
                        .iter()
                        .all(|work| work.kv_slots.len() == work.num_tokens)
                );
            }
        }
    }

    #[test]
    fn decode_work_has_priority_over_prefill_when_budget_is_tight() {
        let mut test_config = config();
        test_config.max_batch_tokens = 1;
        let metrics = Arc::new(Metrics::default());
        let backend = DeterministicBackend::new("test-model");
        let mut worker = EngineWorker::new(backend, test_config, metrics);
        let (decode_output, _decode_receiver) = mpsc::channel(8);
        let decode = request(3);
        let decode_id = decode.id;
        worker.add_request(decode, decode_output);
        let (prefill_output, _prefill_receiver) = mpsc::channel(8);
        let prefill = request(3);
        let prefill_id = prefill.id;
        worker.add_request(prefill, prefill_output);
        worker.admit_waiting();
        let decode_state = worker.requests.get_mut(&decode_id).unwrap();
        decode_state.computed_tokens = decode_state.prompt_tokens;
        decode_state.generated_tokens = 1;

        let batch = worker.build_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].request_id, decode_id);
        assert!(!batch.iter().any(|work| work.request_id == prefill_id));
    }
}
