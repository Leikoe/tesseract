use std::{
    collections::{HashMap, HashSet, VecDeque},
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

use crate::model::{IncrementalDecoder, Model};
use crate::{config::EngineConfig, metrics::Metrics};

use super::{
    ExecutionOutput, FinishReason, ForwardBatch, ForwardPhase, ForwardSequence, GenerateRequest,
    GeneratedTokens, GenerationEvent, GenerationParams, ModelExecutor, Position, RequestId,
    SamplingInput, TokenId, Usage, kv::KvSlots, recurrent::RecurrentSlots,
};

#[derive(Debug, Error)]
pub enum EngineSpawnError {
    #[error("failed to spawn engine thread: {0}")]
    Thread(#[from] std::io::Error),
    #[error("failed to initialize inference executor: {0}")]
    ExecutorInitialization(String),
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
    pub fn spawn<B: ModelExecutor + Send>(
        model_id: impl Into<Arc<str>>,
        executor: B,
        config: EngineConfig,
        command_capacity: usize,
        metrics: Arc<Metrics>,
    ) -> Result<Self, EngineSpawnError> {
        Self::spawn_with_factory(
            model_id,
            move || Ok::<B, Infallible>(executor),
            config,
            command_capacity,
            metrics,
        )
    }

    /// Builds the executor on the dedicated engine thread, then returns only
    /// after model loading and executor initialization have succeeded.
    pub fn spawn_with_factory<B, F, E>(
        model_id: impl Into<Arc<str>>,
        factory: F,
        config: EngineConfig,
        command_capacity: usize,
        metrics: Arc<Metrics>,
    ) -> Result<Self, EngineSpawnError>
    where
        B: ModelExecutor,
        F: FnOnce() -> Result<B, E> + Send + 'static,
        E: Display,
    {
        let model_id = model_id.into();
        let output_buffer = config.output_buffer;
        let admission_limit = config.max_pending + config.max_running;
        let (commands_tx, commands_rx) = mpsc::channel(command_capacity);
        let worker_metrics = Arc::clone(&metrics);
        let (initialized_tx, initialized_rx) = std::sync::mpsc::sync_channel(1);
        thread::Builder::new()
            .name("tesseract-engine".into())
            .spawn(move || match factory() {
                Ok(executor) => match EngineWorker::try_new(executor, config, worker_metrics) {
                    Ok(worker) => {
                        let model = Arc::clone(&worker.model);
                        if initialized_tx.send(Ok(model)).is_ok() {
                            worker.run(commands_rx);
                        }
                    }
                    Err(error) => {
                        let _ = initialized_tx.send(Err(error));
                    }
                },
                Err(error) => {
                    let _ = initialized_tx.send(Err(error.to_string()));
                }
            })?;
        let model = initialized_rx
            .recv()
            .map_err(|_| EngineSpawnError::InitializationChannelClosed)?
            .map_err(EngineSpawnError::ExecutorInitialization)?;
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
    params: GenerationParams,
    output: mpsc::Sender<GenerationEvent>,
    prompt: Vec<TokenId>,
    generated: Vec<TokenId>,
    decoder: Box<dyn IncrementalDecoder>,
    rng: SplitMix64,
    computed_tokens: usize,
    pending_text: String,
    started_at: Instant,
    last_token_at: Option<Instant>,
}

impl RequestState {
    fn target_tokens(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }

    fn reservation_tokens(&self) -> usize {
        self.prompt.len() + self.params.max_tokens
    }

    fn token_range(&self, range: std::ops::Range<usize>) -> Vec<TokenId> {
        self.prompt
            .iter()
            .chain(&self.generated)
            .copied()
            .skip(range.start)
            .take(range.len())
            .collect()
    }

    fn consume_text(&mut self, text: &str) -> (String, bool) {
        self.pending_text.push_str(text);
        if self.params.stop.is_empty() {
            return (std::mem::take(&mut self.pending_text), false);
        }

        let stop_at = self
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
                    && self.params.stop.iter().any(|stop| {
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

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn unit_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

struct EngineWorker<B> {
    executor: B,
    model: Arc<dyn Model>,
    config: EngineConfig,
    metrics: Arc<Metrics>,
    kv: KvSlots,
    recurrent: Option<RecurrentSlots>,
    requests: HashMap<RequestId, RequestState>,
    waiting: VecDeque<RequestId>,
    running: VecDeque<RequestId>,
    in_flight_requests: HashSet<RequestId>,
    deferred_cancellations: HashSet<RequestId>,
    shutting_down: bool,
    shutdown_ack: Option<oneshot::Sender<()>>,
}

impl<B: ModelExecutor> EngineWorker<B> {
    #[cfg(test)]
    fn new(executor: B, config: EngineConfig, metrics: Arc<Metrics>) -> Self {
        Self::try_new(executor, config, metrics)
            .expect("test executor must provide compatible state")
    }

    fn try_new(executor: B, config: EngineConfig, metrics: Arc<Metrics>) -> Result<Self, String> {
        let state_schema = executor.state_schema();
        let physical_capacity = state_schema
            .flat_kv_capacity()
            .ok_or_else(|| "executor state schema has no flat-KV group".to_owned())?;
        if config.kv_capacity_tokens > physical_capacity {
            return Err(format!(
                "engine requests {} KV slots but executor arena {} provides {physical_capacity}",
                config.kv_capacity_tokens,
                state_schema.arena_id()
            ));
        }
        if let Some(physical_capacity) = state_schema.recurrent_capacity()
            && config.max_running > physical_capacity
        {
            return Err(format!(
                "engine requests {} recurrent-state slots but executor arena {} provides {physical_capacity}",
                config.max_running,
                state_schema.arena_id()
            ));
        }
        let model = executor.model();
        let kv = KvSlots::new(state_schema.arena_id(), config.kv_capacity_tokens);
        let recurrent = state_schema.recurrent_capacity().map(RecurrentSlots::new);
        Ok(Self {
            executor,
            model,
            kv,
            recurrent,
            config,
            metrics,
            requests: HashMap::new(),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            in_flight_requests: HashSet::new(),
            deferred_cancellations: HashSet::new(),
            shutting_down: false,
            shutdown_ack: None,
        })
    }

    fn run(mut self, mut commands: mpsc::Receiver<Command>) {
        self.metrics
            .add_execution(self.executor.take_execution_stats());
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
            self.execute_batch(batch, &mut commands);
            self.update_gauges();
        }

        self.metrics.set_ready(false);
        if let Err(error) = self.executor.shutdown() {
            tracing::error!(%error, "executor shutdown failed");
        }
        if let Some(ack) = self.shutdown_ack.take() {
            let _ = ack.send(());
        }
    }

    fn execute_batch(&mut self, batch: ForwardBatch, commands: &mut mpsc::Receiver<Command>) {
        let ticket = match self.executor.submit(&batch) {
            Ok(ticket) => ticket,
            Err(error) => {
                self.fail_batch(&batch, error.to_string());
                return;
            }
        };
        self.in_flight_requests
            .extend(batch.sequences().iter().map(ForwardSequence::request_id));

        loop {
            self.receive_commands(commands);
            self.prune_disconnected();
            match self.executor.poll(ticket.completion()) {
                Ok(Some(output)) => {
                    self.finish_execution(batch, output);
                    break;
                }
                Ok(None) => thread::yield_now(),
                Err(error) => {
                    self.clear_in_flight();
                    self.fail_batch(&batch, error.to_string());
                    break;
                }
            }
        }
        self.metrics
            .add_execution(self.executor.take_execution_stats());
    }

    fn finish_execution(&mut self, batch: ForwardBatch, output: ExecutionOutput) {
        self.clear_in_flight();
        let cancelled = std::mem::take(&mut self.deferred_cancellations);
        for request_id in cancelled {
            self.finish_request(request_id, FinishReason::Cancelled);
        }
        match output {
            ExecutionOutput::Generation { requests } => self.apply_step(&batch, requests),
        }
    }

    fn clear_in_flight(&mut self) {
        self.in_flight_requests.clear();
    }

    fn fail_batch(&mut self, batch: &ForwardBatch, message: String) {
        let cancelled = std::mem::take(&mut self.deferred_cancellations);
        for sequence in batch.sequences() {
            let request_id = sequence.request_id();
            if cancelled.contains(&request_id) {
                self.finish_request(request_id, FinishReason::Cancelled);
            } else {
                self.fail_request(request_id, message.clone());
            }
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
        let started = Instant::now();
        let prompt = match self.model.encode(&request.prompt) {
            Ok(prompt) => prompt.into_iter().map(TokenId::new).collect::<Vec<_>>(),
            Err(error) => {
                let _ = output.try_send(GenerationEvent::Failed {
                    message: error.to_string(),
                });
                self.metrics.request_failed();
                return;
            }
        };
        if prompt.is_empty() {
            let _ = output.try_send(GenerationEvent::Failed {
                message: "the rendered prompt is empty".into(),
            });
            self.metrics.request_failed();
            return;
        }
        let prompt_tokens = prompt.len();
        let total = prompt_tokens + request.params.max_tokens;
        if total > self.config.max_sequence_length || total > self.kv.capacity() {
            let _ = output.try_send(GenerationEvent::Failed {
                message: format!(
                    "prompt plus max_tokens ({total}) exceeds the configured capacity"
                ),
            });
            self.metrics.request_failed();
            return;
        }

        self.metrics.request_started();
        self.metrics.add_prompt_tokens(prompt_tokens);
        tracing::debug!(
            request_id = %id,
            prompt_tokens,
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "prompt tokenized"
        );
        let decoder = self.model.decoder();
        let rng = SplitMix64::new(request.params.seed);
        self.requests.insert(
            id,
            RequestState {
                params: request.params,
                output,
                prompt,
                generated: Vec::with_capacity(total - prompt_tokens),
                decoder,
                rng,
                computed_tokens: 0,
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
            if !self.kv.reserve(id, state.reservation_tokens()) {
                self.waiting.push_back(id);
                continue;
            }
            let has_recurrent_slot = self
                .recurrent
                .as_mut()
                .is_none_or(|slots| slots.allocate(id).is_some());
            if has_recurrent_slot {
                self.running.push_back(id);
            } else {
                self.kv.release(id);
                self.waiting.push_back(id);
            }
        }
        self.update_gauges();
    }

    fn build_batch(&mut self) -> ForwardBatch {
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
                let is_decode = state.computed_tokens >= state.prompt.len();
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
                let context_slots = self.kv.request_slots(*id).to_vec();
                let state = self
                    .requests
                    .get_mut(id)
                    .expect("scheduled request must remain present");
                let end = state
                    .computed_tokens
                    .checked_add(num_tokens)
                    .expect("validated sequence length must not overflow");
                let token_ids = state.token_range(state.computed_tokens..end);
                let sampling = (end == state.target_tokens()).then(|| {
                    let random_sample = if state.params.temperature == 0.0 {
                        0.0
                    } else {
                        state.rng.unit_f64()
                    };
                    SamplingInput::try_new(
                        state.params.temperature,
                        state.params.top_p,
                        random_sample,
                    )
                    .expect("validated generation parameters must form sampling input")
                });
                let sequence = ForwardSequence::try_new(
                    *id,
                    if is_decode {
                        ForwardPhase::Decode
                    } else {
                        ForwardPhase::Prefill
                    },
                    Position::new(state.computed_tokens),
                    token_ids,
                    kv_slots,
                    context_slots,
                    sampling,
                )
                .expect("scheduler must construct a valid forward sequence");
                let sequence = match self.recurrent.as_ref() {
                    Some(slots) => sequence.with_recurrent_slot(
                        slots
                            .get(*id)
                            .expect("running hybrid request must own recurrent state"),
                    ),
                    None => sequence,
                };
                batch.push(sequence);
                budget -= num_tokens;
            }
        }
        // Preserve decode-before-prefill priority while rotating peers of the
        // same phase so a request at the front cannot monopolize a tight
        // token budget across engine iterations.
        if self.running.len() > 1 {
            self.running.rotate_left(1);
        }
        ForwardBatch::try_from_sequences(self.kv.arena_id(), batch)
            .expect("scheduler must construct a valid non-aliasing batch")
    }

    fn apply_step(&mut self, batch: &ForwardBatch, outputs: Vec<GeneratedTokens>) {
        let expected_outputs = batch
            .sequences()
            .iter()
            .filter(|sequence| sequence.should_sample())
            .map(ForwardSequence::request_id)
            .collect::<HashSet<_>>();
        let mut output_map = HashMap::with_capacity(outputs.len());
        let malformed = outputs.into_iter().any(|output| {
            let request_id = output.request_id();
            !expected_outputs.contains(&request_id)
                || output_map.insert(request_id, output).is_some()
        });
        if malformed {
            let request_ids = batch
                .sequences()
                .iter()
                .map(ForwardSequence::request_id)
                .collect::<Vec<_>>();
            for request_id in request_ids {
                self.fail_request(
                    request_id,
                    "executor returned a duplicate or unexpected sampled output".into(),
                );
            }
            return;
        }
        let mut finish = Vec::new();
        let mut failed = Vec::new();

        for sequence in batch.sequences() {
            let Some(state) = self.requests.get_mut(&sequence.request_id()) else {
                continue;
            };
            state.computed_tokens += sequence.num_tokens();
            if !sequence.should_sample() {
                continue;
            }
            let Some(output) = output_map.remove(&sequence.request_id()) else {
                failed.push((
                    sequence.request_id(),
                    "executor omitted output for a sampled request".to_string(),
                ));
                continue;
            };

            let token_ids = output.into_token_ids();
            if token_ids.is_empty() {
                failed.push((
                    sequence.request_id(),
                    "executor returned no progress for a sampled request".into(),
                ));
                continue;
            }

            for token_id in token_ids {
                if state.generated.len() >= state.params.max_tokens {
                    finish.push((sequence.request_id(), FinishReason::Length));
                    break;
                }
                let text = match state.decoder.push(token_id.get()) {
                    Ok(text) => text,
                    Err(error) => {
                        failed.push((sequence.request_id(), error.to_string()));
                        break;
                    }
                };
                state.generated.push(token_id);
                self.metrics.token_generated();
                let now = Instant::now();
                match state.last_token_at {
                    Some(previous) => self.metrics.observe_inter_token(now - previous),
                    None => self.metrics.observe_ttft(now - state.started_at),
                }
                state.last_token_at = Some(now);
                let (text, hit_stop) = state.consume_text(&text);
                if !text.is_empty()
                    && state
                        .output
                        .try_send(GenerationEvent::Delta {
                            text,
                            token_id: Some(token_id.get()),
                        })
                        .is_err()
                {
                    finish.push((sequence.request_id(), FinishReason::Cancelled));
                    break;
                }

                let reason = if hit_stop || self.model.eos_token_ids().contains(&token_id.get()) {
                    Some(FinishReason::Stop)
                } else if state.generated.len() >= state.params.max_tokens {
                    Some(FinishReason::Length)
                } else {
                    None
                };
                if let Some(reason) = reason {
                    finish.push((sequence.request_id(), reason));
                    break;
                }
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
        if let Some(recurrent) = &mut self.recurrent {
            recurrent.release(id);
        }
        let elapsed = state.started_at.elapsed();
        self.metrics.observe_request_duration(elapsed);

        if !state.pending_text.is_empty() && reason != FinishReason::Cancelled {
            let _ = state.output.try_send(GenerationEvent::Delta {
                text: std::mem::take(&mut state.pending_text),
                token_id: None,
            });
        }
        let usage = Usage::new(state.prompt.len(), state.generated.len());
        let _ = state
            .output
            .try_send(GenerationEvent::Finished { reason, usage });
        match reason {
            FinishReason::Cancelled => self.metrics.request_cancelled(),
            FinishReason::Error => self.metrics.request_failed(),
            FinishReason::Stop | FinishReason::Length => self.metrics.request_completed(),
        }
        tracing::info!(
            request_id = %id,
            finish_reason = ?reason,
            prompt_tokens = state.prompt.len(),
            generated_tokens = state.generated.len(),
            latency_ms = elapsed.as_secs_f64() * 1_000.0,
            "request finished"
        );
        self.update_gauges();
    }

    fn fail_request(&mut self, id: RequestId, message: String) {
        let Some(state) = self.requests.remove(&id) else {
            return;
        };
        self.waiting.retain(|candidate| *candidate != id);
        self.running.retain(|candidate| *candidate != id);
        self.kv.release(id);
        if let Some(recurrent) = &mut self.recurrent {
            recurrent.release(id);
        }
        self.metrics.request_failed();
        tracing::warn!(
            request_id = %id,
            error = %message,
            latency_ms = state.started_at.elapsed().as_secs_f64() * 1_000.0,
            "request failed"
        );
        let _ = state.output.try_send(GenerationEvent::Failed { message });
        self.update_gauges();
    }

    fn cancel_request(&mut self, id: RequestId) {
        if self.in_flight_requests.contains(&id) {
            self.waiting.retain(|candidate| *candidate != id);
            self.running.retain(|candidate| *candidate != id);
            self.deferred_cancellations.insert(id);
        } else {
            self.finish_request(id, FinishReason::Cancelled);
        }
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
    use crate::engine::{GenerationParams, QueryRow, testing::DeterministicExecutor};
    use proptest::prelude::*;
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

    fn worker_ready_to_sample(
        max_tokens: usize,
    ) -> (
        EngineWorker<DeterministicExecutor>,
        ForwardBatch,
        mpsc::Receiver<GenerationEvent>,
        RequestId,
    ) {
        let metrics = Arc::new(Metrics::default());
        let executor = DeterministicExecutor::new();
        let mut worker = EngineWorker::new(executor, config(), metrics);
        let (output, receiver) = mpsc::channel(32);
        let request = request(max_tokens);
        let request_id = request.id;
        worker.add_request(request, output);
        worker.admit_waiting();
        loop {
            let batch = worker.build_batch();
            if batch.iter().any(ForwardSequence::should_sample) {
                return (worker, batch, receiver, request_id);
            }
            worker.apply_step(&batch, Vec::new());
        }
    }

    #[test]
    fn startup_rejects_an_executor_with_insufficient_physical_state() {
        let metrics = Arc::new(Metrics::default());
        let result = EngineHandle::spawn(
            "test-model",
            DeterministicExecutor::new().with_state_capacity(8),
            config(),
            8,
            metrics,
        );
        assert!(matches!(
            result,
            Err(EngineSpawnError::ExecutorInitialization(message))
                if message.contains("provides 8")
        ));
    }

    #[test]
    fn startup_rejects_insufficient_recurrent_state() {
        let metrics = Arc::new(Metrics::default());
        let result = EngineHandle::spawn(
            "test-model",
            DeterministicExecutor::new().with_hybrid_state(64, 1),
            config(),
            8,
            metrics,
        );
        assert!(matches!(
            result,
            Err(EngineSpawnError::ExecutorInitialization(message))
                if message.contains("recurrent-state slots") && message.contains("provides 1")
        ));
    }

    #[test]
    fn hybrid_requests_receive_exclusive_recurrent_slots() {
        let metrics = Arc::new(Metrics::default());
        let executor = DeterministicExecutor::new().with_hybrid_state(64, 2);
        let mut worker = EngineWorker::new(executor, config(), metrics);
        for _ in 0..2 {
            let (output, _receiver) = mpsc::channel(8);
            worker.add_request(request(1), output);
        }
        worker.admit_waiting();
        let batch = worker.build_batch();
        let slots = batch
            .iter()
            .map(|sequence| sequence.recurrent_slot().unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(slots.len(), 2);

        let released = batch[0].request_id();
        worker.finish_request(released, FinishReason::Cancelled);
        assert_eq!(worker.recurrent.as_ref().unwrap().get(released), None);
    }

    #[tokio::test]
    async fn produces_deltas_and_length_finish() {
        let metrics = Arc::new(Metrics::default());
        let engine = EngineHandle::spawn(
            "test-model",
            DeterministicExecutor::new(),
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
            "test-model",
            DeterministicExecutor::new(),
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
            "test-model",
            DeterministicExecutor::new(),
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
        let engine = EngineHandle::spawn(
            "test-model",
            DeterministicExecutor::new(),
            bounded,
            8,
            metrics,
        )
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
            "test-model",
            DeterministicExecutor::new(),
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
            "test-model",
            DeterministicExecutor::new().with_submission_delay(Duration::from_millis(25)),
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

    #[tokio::test]
    async fn executor_failure_releases_state_and_the_engine_keeps_serving() {
        let metrics = Arc::new(Metrics::default());
        let mut single = config();
        single.max_running = 1;
        let engine = EngineHandle::spawn(
            "test-model",
            DeterministicExecutor::new().failing_next_submission(),
            single,
            8,
            Arc::clone(&metrics),
        )
        .unwrap();

        let mut failed = engine.try_generate(request(1)).unwrap();
        assert!(matches!(
            failed.recv().await,
            Some(GenerationEvent::Failed { message }) if message.contains("injected submission failure")
        ));

        let mut recovered = engine.try_generate(request(1)).unwrap();
        let terminal = tokio::time::timeout(Duration::from_secs(1), async {
            while let Some(event) = recovered.recv().await {
                if matches!(event, GenerationEvent::Finished { .. }) {
                    return event;
                }
            }
            panic!("recovery request closed without a terminal event");
        })
        .await
        .unwrap();
        assert!(matches!(
            terminal,
            GenerationEvent::Finished {
                reason: FinishReason::Length,
                ..
            }
        ));
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if metrics.prometheus().contains("tesseract_kv_tokens_used 0") {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn duplicate_executor_outputs_fail_without_leaking_request_state() {
        let metrics = Arc::new(Metrics::default());
        let executor = DeterministicExecutor::new();
        let mut worker = EngineWorker::new(executor, config(), metrics);
        let (output, _receiver) = mpsc::channel(8);
        let request = request(1);
        let request_id = request.id;
        worker.add_request(request, output);
        worker.admit_waiting();

        let first = worker.build_batch();
        assert!(!first[0].should_sample());
        worker.apply_step(&first, Vec::new());
        let sampled = worker.build_batch();
        assert!(sampled[0].should_sample());
        let duplicate = GeneratedTokens::one(request_id, TokenId::new(1000));
        worker.apply_step(&sampled, vec![duplicate.clone(), duplicate]);

        assert!(worker.requests.is_empty());
        assert_eq!(worker.kv.used(), 0);
    }

    #[test]
    fn empty_sample_output_fails_instead_of_stalling() {
        let (mut worker, batch, _receiver, request_id) = worker_ready_to_sample(2);
        worker.apply_step(&batch, vec![GeneratedTokens::new(request_id, Vec::new())]);
        assert!(worker.requests.is_empty());
        assert_eq!(worker.kv.used(), 0);
    }

    #[test]
    fn variable_output_is_truncated_at_the_request_length() {
        let (mut worker, batch, mut receiver, request_id) = worker_ready_to_sample(2);
        worker.apply_step(
            &batch,
            vec![GeneratedTokens::new(
                request_id,
                vec![TokenId::new(1000), TokenId::new(1001), TokenId::new(1002)],
            )],
        );

        assert!(worker.requests.is_empty());
        assert_eq!(worker.kv.used(), 0);
        let events = std::iter::from_fn(|| receiver.try_recv().ok()).collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, GenerationEvent::Delta { .. }))
                .count(),
            2
        );
        assert!(events.iter().any(|event| matches!(
            event,
            GenerationEvent::Finished {
                reason: FinishReason::Length,
                usage,
            } if usage.completion_tokens == 2
        )));
    }

    #[test]
    fn variable_output_stops_at_the_first_terminal_token() {
        let (mut worker, batch, mut receiver, request_id) = worker_ready_to_sample(4);
        worker.requests.get_mut(&request_id).unwrap().params.stop = vec!["token1".into()];
        worker.apply_step(
            &batch,
            vec![GeneratedTokens::new(
                request_id,
                vec![TokenId::new(1000), TokenId::new(1001), TokenId::new(1002)],
            )],
        );

        assert!(worker.requests.is_empty());
        let events = std::iter::from_fn(|| receiver.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            GenerationEvent::Finished {
                reason: FinishReason::Stop,
                usage,
            } if usage.completion_tokens == 2
        )));
    }

    #[test]
    fn cancellation_defers_kv_reclamation_until_ticket_completion() {
        let metrics = Arc::new(Metrics::default());
        let executor = DeterministicExecutor::new();
        let mut worker = EngineWorker::new(executor, config(), metrics);
        let (output, _receiver) = mpsc::channel(8);
        let request = request(1);
        let request_id = request.id;
        worker.add_request(request, output);
        worker.admit_waiting();

        let batch = worker.build_batch();
        let ticket = worker.executor.submit(&batch).unwrap();
        worker.in_flight_requests.insert(request_id);
        worker.cancel_request(request_id);

        assert!(worker.requests.contains_key(&request_id));
        assert!(worker.deferred_cancellations.contains(&request_id));
        assert!(!worker.running.contains(&request_id));
        assert!(worker.kv.used() > 0);

        let output = worker.executor.poll(ticket.completion()).unwrap().unwrap();
        worker.finish_execution(batch, output);

        assert!(!worker.requests.contains_key(&request_id));
        assert!(worker.in_flight_requests.is_empty());
        assert!(worker.deferred_cancellations.is_empty());
        assert_eq!(worker.kv.used(), 0);
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
                let executor = DeterministicExecutor::new();
                let mut worker = EngineWorker::new(executor, test_config, metrics);
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
                assert!(batch.iter().map(ForwardSequence::num_tokens).sum::<usize>() <= budget);
                assert!(batch.iter().all(|sequence| {
                    sequence.phase() == ForwardPhase::Decode || sequence.num_tokens() <= chunk
                }));
                let slots: Vec<_> = batch
                    .iter()
                    .flat_map(|sequence| sequence.kv_slots().iter().copied())
                    .collect();
                assert_eq!(
                    slots.iter().copied().collect::<HashSet<_>>().len(),
                    slots.len()
                );
                assert!(
                    batch
                        .iter()
                        .all(|sequence| sequence.kv_slots().len() == sequence.num_tokens())
                );
            }
        }
    }

    #[test]
    fn decode_work_has_priority_over_prefill_when_budget_is_tight() {
        let mut test_config = config();
        test_config.max_batch_tokens = 1;
        let metrics = Arc::new(Metrics::default());
        let executor = DeterministicExecutor::new();
        let mut worker = EngineWorker::new(executor, test_config, metrics);
        let (decode_output, _decode_receiver) = mpsc::channel(8);
        let decode = request(3);
        let decode_id = decode.id;
        worker.add_request(decode, decode_output);
        let (prefill_output, _prefill_receiver) = mpsc::channel(8);
        let prefill = request(3);
        let prefill_id = prefill.id;
        worker.add_request(prefill, prefill_output);
        worker.admit_waiting();
        let prompt_len = worker.requests[&decode_id].prompt.len();
        worker
            .kv
            .allocate(decode_id, prompt_len)
            .expect("decode fixture must materialize its prompt KV");
        let decode_state = worker.requests.get_mut(&decode_id).unwrap();
        decode_state.computed_tokens = prompt_len;
        decode_state.generated.push(TokenId::new(1000));

        let batch = worker.build_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].request_id(), decode_id);
        assert!(
            !batch
                .iter()
                .any(|sequence| sequence.request_id() == prefill_id)
        );
    }

    #[test]
    fn equal_phase_requests_rotate_under_a_tight_budget() {
        let mut test_config = config();
        test_config.max_batch_tokens = 1;
        test_config.prefill_chunk_tokens = 1;
        let metrics = Arc::new(Metrics::default());
        let executor = DeterministicExecutor::new();
        let mut worker = EngineWorker::new(executor, test_config, metrics);
        let (first_output, _first_receiver) = mpsc::channel(8);
        let first = request(3);
        let first_id = first.id;
        worker.add_request(first, first_output);
        let (second_output, _second_receiver) = mpsc::channel(8);
        let second = request(3);
        let second_id = second.id;
        worker.add_request(second, second_output);
        worker.admit_waiting();

        let first_batch = worker.build_batch();
        let second_batch = worker.build_batch();

        assert_eq!(first_batch.len(), 1);
        assert_eq!(first_batch[0].request_id(), first_id);
        assert_eq!(second_batch.len(), 1);
        assert_eq!(second_batch[0].request_id(), second_id);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn variable_generation_never_commits_past_length(
            max_tokens in 1usize..12,
            extra_tokens in 0usize..8,
        ) {
            let (mut worker, batch, mut receiver, request_id) =
                worker_ready_to_sample(max_tokens);
            let accepted = (0..max_tokens + extra_tokens)
                .map(|offset| TokenId::new(1000 + offset as u32))
                .collect();
            worker.apply_step(&batch, vec![GeneratedTokens::new(request_id, accepted)]);

            prop_assert!(worker.requests.is_empty());
            prop_assert_eq!(worker.kv.used(), 0);
            let events = std::iter::from_fn(|| receiver.try_recv().ok()).collect::<Vec<_>>();
            let deltas = events
                .iter()
                .filter(|event| matches!(event, GenerationEvent::Delta { .. }))
                .count();
            prop_assert_eq!(deltas, max_tokens);
            let usage_is_exact = events.iter().any(|event| matches!(
                event,
                GenerationEvent::Finished {
                    reason: FinishReason::Length,
                    usage,
                } if usage.completion_tokens == max_tokens
            ));
            prop_assert!(usage_is_exact);
        }

        #[test]
        fn arbitrary_scheduler_runs_preserve_batch_and_kv_invariants(
            requests in prop::collection::vec((1usize..16, 1usize..8), 1..7),
            token_budget in 1usize..33,
            prefill_chunk in 1usize..17,
        ) {
            let max_running = requests.len();
            let token_budget = token_budget.max(max_running);
            let mut test_config = config();
            test_config.max_running = max_running;
            test_config.max_batch_tokens = token_budget;
            test_config.prefill_chunk_tokens = prefill_chunk;
            test_config.max_sequence_length = 64;
            test_config.kv_capacity_tokens = 512;
            test_config.output_buffer = 128;

            let metrics = Arc::new(Metrics::default());
            let executor = DeterministicExecutor::new();
            let mut worker = EngineWorker::new(executor, test_config, metrics);
            let mut receivers = Vec::with_capacity(requests.len());
            let mut maximum_steps = 0usize;
            for (prompt_tokens, max_tokens) in requests {
                maximum_steps += prompt_tokens + max_tokens;
                let (output, receiver) = mpsc::channel(128);
                receivers.push(receiver);
                let mut generated = request(max_tokens);
                generated.prompt = (0..prompt_tokens)
                    .map(|index| format!("token{index}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                worker.add_request(generated, output);
            }
            worker.admit_waiting();

            let mut steps = 0usize;
            while !worker.requests.is_empty() {
                steps += 1;
                prop_assert!(steps <= maximum_steps, "scheduler stopped making progress");
                let before: HashMap<_, _> = worker
                    .requests
                    .iter()
                    .map(|(id, state)| (*id, state.computed_tokens))
                    .collect();
                let batch = worker.build_batch();
                prop_assert!(!batch.is_empty());
                prop_assert!(batch.num_tokens() <= token_budget);
                prop_assert_eq!(
                    batch.query_start_offsets().last().copied().map(QueryRow::get),
                    Some(batch.num_tokens())
                );
                let phases_are_partitioned = batch.sequences().windows(2).all(|pair| {
                    pair[0].phase() != ForwardPhase::Decode
                        || pair[1].phase() == ForwardPhase::Decode
                });
                prop_assert!(phases_are_partitioned);
                let work_is_valid = batch.sequences().iter().all(|sequence| {
                    sequence.kv_slots().len() == sequence.num_tokens()
                        && sequence.token_ids().len() == sequence.num_tokens()
                        && sequence.context_slots().len()
                            == sequence.position().get() + sequence.num_tokens()
                        && sequence.context_slots().ends_with(sequence.kv_slots())
                        && match sequence.phase() {
                            ForwardPhase::Prefill => sequence.num_tokens() <= prefill_chunk,
                            ForwardPhase::Decode => {
                                sequence.num_tokens() == 1 && sequence.should_sample()
                            }
                        }
                });
                prop_assert!(work_is_valid);
                let slots: HashSet<_> = batch
                    .sequences()
                    .iter()
                    .flat_map(|sequence| sequence.kv_slots().iter().copied())
                    .collect();
                prop_assert_eq!(slots.len(), batch.num_tokens());

                let ticket = worker.executor.submit(&batch).unwrap();
                let output = worker
                    .executor
                    .poll(ticket.completion())
                    .unwrap()
                    .expect("deterministic executor completes synchronously");
                let ExecutionOutput::Generation { requests } = output;
                prop_assert_eq!(
                    requests.len(),
                    batch
                        .sequences()
                        .iter()
                        .filter(|sequence| sequence.should_sample())
                        .count()
                );
                worker.apply_step(&batch, requests);
                prop_assert!(worker.kv.used() <= 512);
                for (id, state) in &worker.requests {
                    prop_assert!(state.computed_tokens >= before[id]);
                }
            }
            prop_assert_eq!(worker.kv.used(), 0);
        }
    }
}
