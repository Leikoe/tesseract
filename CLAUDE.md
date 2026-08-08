# Project Instructions and User Message Log

Every user message in this project must be appended to this file verbatim.

## User Messages (verbatim)

```text
dispatch subagents to analyze references/{vllm,sglang}. You will need an in-depth report of how each works.
```

```text
could you search sglang blogs or whatever online about their adoption of flat kv cache recently ? (july 2026?)
```

```text
I think you should make a document with everything we learned today for future reference
```

```text
We want to make an inference server which will be made for maximum performance and production from day 1.
```

```text
we can start by supporting bf16 only. what stack should we base it on ? i'd say tokio, tower, axum ? for the actual compute, we can use https://github.com/nvlabs/cutile-rs ?
```

```text
I will give you an A100 box. you can ssh to it and test your code there when needed. HOWEVER, it is a spot instance and might die at any point, so always use the git repo to save your work, and pull on the box (scp anything important back to local). Also, each message I send you needs to be saved verbatim inside CLAUDE.md (and this instruction too!).
```

```text
I say you use llama 3.2 1B Instruct for now. you have a .env with the hf token to hf download it. you can copy that .env to the A100 machine it's ok. "ssh ubuntu@216.81.245.246"
```

```text
I made it public, retry
```
