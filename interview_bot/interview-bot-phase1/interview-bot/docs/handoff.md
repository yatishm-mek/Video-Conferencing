# Interview bot — full context handoff (as of 2026-11-09 20:30 IST)

Read this first in a new chat. It replaces the 2026-11-07 handoff. Older Phase 1 details: `claude/build-log.md`.

## 1. What we are building
An in-house, open-source replacement for Retell AI for the company's AI interview website (task from Hardhik's TL). A real-time voice bot reads the candidate's resume + JD, prepares tailored questions, interviews by voice, understands what the candidate means, asks follow-ups / clarifications, flags things for a human, scores each answer, and (Phase 6, not done) writes a recruiter report. **A human makes the final hiring decision.**

## 2. Hardhik's rules (keep applying these)
- **Deploy on CPU.** Clients may not have GPU servers. Develop on the laptop GPU, but everything must work on CPU.
- **No hard-coding of meaning.** "If we put hardcoding, the candidate will say other words." The LLM must understand varied wording. Hard-code only audio chunking, latency and timing.
- **The LLM must not invent anything**: no new words, facts or names. It acts as the interviewer, never as the candidate.
- **Storage:** store nothing outside `logs/` and `data/`.
- **Transcript fixes:** sound-alike only (Laura -> LoRA). Never change numbers, never correct the candidate's real mistakes. LLM fixes are shown in the log as changed by the LLM.
- **Scoring:** content only (never accent or grammar), at medium depth. "Suspicious" is advisory.
- **Tone:** polite closings ("Thank you for your time…", never "Goodbye").
- **Never skip a question without the candidate's permission.** Never leave an unfinished answer without asking "anything more?".
- **Interview questions vs HR:** questions about the interview itself are answered as the AI interviewer from facts. Only HR matters (salary, leave, benefits, offer, joining, results, next steps) go to "the recruiter can help with that". When pushing back, say "this is an AI interview".
- **Measure accuracy honestly.** Never tune the prompt to eval sentences (he objected: "why are you hardcoding this??"). Use held-out sets.
- **How he likes to work:**
  - Concise replies and exact commands.
  - Copy files straight to the laptop.
  - Back up working versions, and keep old engines for A/B.
  - When he says "check eval", read the latest session in `logs/server.log`, list every bot mistake, and give scores.

## 3. Machine and how to run
- **Laptop:** Windows, Python 3.11.9, RTX 5050 Laptop 8 GB, AMD 780M iGPU (ignored by Ollama).
- **Paths:** project `D:\In_house_retellai\interview-bot-phase1\interview-bot`. The venv is ONE LEVEL UP: `D:\In_house_retellai\.venv`.
- **Activate:** `D:\In_house_retellai\.venv\Scripts\Activate.ps1` (`.\.venv\...` does NOT exist in the project folder).
- **Ollama 0.40.15:**
  - Live model: `gemma4:e4b`.
  - Prep model: `qwen3.5:9b` (think off).
  - If it crashes with "CUDA error: shared object initialization failed": quit the tray app, run `taskkill /F /IM ollama.exe /T; taskkill /F /IM "ollama app.exe" /T`, then start Ollama from the Start menu and check `ollama list`. Running `ollama serve` in a window and then closing it = "connection refused".
- **Laptop `.env`:**
  ```
  INTERVIEW_PLAN_PATH=data/candidates/rithika/plan.json
  RESUME_MAX_AGE_SECS=0
  LIVE_LLM_MODEL=gemma4:e4b
  NATURAL_MODE=two_step
  ```
  - `LABEL_FORMAT` is not set, so it defaults to json; compact exists but is not yet measured.
  - Remove `RESUME_MAX_AGE_SECS=0` before production.
- **Run:** `python -m interview_bot.server`, open the page in Chrome, pick the Realtek mic (not "EShare Audio").
  - Code changes need a server restart.
  - Port busy: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 7860).OwningProcess -Force`.
- **Tests:** `python -m pytest -q` (**374 passing**).
- **Claude's cloud mirror:** `/home/claude/interview-bot`. Files go to the laptop through the desktop bridge.
  - ALWAYS copy into a NEW `/mnt/user-data/outputs/<name>` folder, commit, then verify md5 on the laptop. Stale staged copies happened before.
- **Backup:** `D:\In_house_retellai\backups\interview-bot_natural-engine-v1_2026-11-08.zip`. The stray file `backups\zivZmPnL` can be deleted.
- **Log tags:** `[turn]` (label -> move -> action, understand time, first speech, spoken text), `[ctrl]`, `[guard]`, `[flag]`, `[score]`, `[stt-fix by LLM]`, `[echo]`, `[latency]`.

## 4. Stack
- Pipecat 1.15.0 with SmallWebRTC, Silero VAD, Smart Turn v3, segmented faster-whisper large-v3-turbo, Kokoro (kokoro-onnx, CPU), and Ollama /api/chat at `http://127.0.0.1:11434`.
- Always use 127.0.0.1, never "localhost": localhost costs ~2 s per new connection on Windows.
- `think=False`. Ollama prefix cache works.
- Note: `prompt_eval_count` always shows the full prompt size, so check `prompt_eval_duration` to see whether the cache was reused.

## 5. Engines (A/B switch `NATURAL_MODE`)
- **`two_step` (current, Hardhik's choice).** Files: `interview_bot/live/two_step.py` + `NaturalController.decide_two_step` in `natural.py`.
  1. **UNDERSTAND call:** the model returns label, complete, sure, asks_too and finished.
     - `LABEL_FORMAT=json`: Ollama grammar, about 29 tokens.
     - `compact`: one line like `answer n y n y`, about 8 tokens. A malformed line triggers a retry with the JSON grammar.
  2. **Code** maps label + state to a move, with all limits and permissions applied in code.
  3. **SAY call** streams 1–2 short sentences for that move. Planned questions are spoken word for word by code.
  - Both calls share ONE system prompt: role, name, number of questions, minutes, the label list, and the first ~100 words of the JD. It is cached. The last 10 turns are sent as chat messages.
  - **Labels (25):** ready, answer, continuing, yes, no, repeat, rephrase, term_meaning, check_meaning, already_answered, decline, dont_know, skip, stop, question, go_back, correction, ai_help, cant_do, upset, audio, off_topic, other, unclear.
  - **Facts given to the model** (the `_ts_facts` function), all true values from settings:
    - it is the AI interviewer, with no personal name
    - it is the one taking the interview now; a recruiter reviews it afterwards
    - question n of N, minutes left of the total
    - about 3 minutes per question; one answer can take up to about 3 minutes
    - the questions are based on the resume + JD
    - it cannot promise results
- **`llm`** (one streaming call does everything) and **`rules`** (the old code-rule engine): kept for A/B. The classic `BOT_ENGINE=classic` engine also still exists.

## 6. Edge cases handled (all have tests)
**Before Q1**
- Any wording of yes ("okay", "why not", "let's go") starts Q1.
- Questions such as name, length or number of questions are answered, then "Are you ready?" (varied wording).
- "Yes, how long will it take?": it answers, then starts. This is the rule for two things in one turn: answer, then act.
- "No" to ready: it offers a moment or to end (as statements, no question back), waits 10 s, then ends politely.
- "Wait, I'm ready" during that goodbye reopens it once.
- A 2nd "no" ends, mentioning the minutes already spent.
- Silence: it asks ready again.

**While answering**
- 5 s of silence: "would you like a moment?"
- If they ask for time: a reminder at 15 s and a check-in at 25 s.
- The silence count resets when they say something real ("yes" to a moment offer doesn't reset it, which avoids the old loop).
- At the end of the silence steps it asks permission, naming the question ("shall we skip question N?"). Only if there is still no reply does it move on ("Since I haven't heard from you…"), flagged.
- Stopped mid-thought (the model's `finished=false`, any wording): it waits 3 s silently, once per set of words, then acts on everything said. It fires on 3–7% of turns, all real pauses.
- Further speech after a short pause cancels the pending reply. A noise-only turn with words pending acts on those words.
- Repeat: 2 times, then a rephrase, then "already repeated" plus the time left, then move on.
- Rephrase / everyday phrase: explained without hints. A technical term: "can't explain terms in the interview".
- Clarifications: at most 3 per interview, then it says it's an interview and how much time is left.
- "So you're asking about X, right?": it confirms.
- Not sure what they meant: "Just to check, do you mean…?", at most 3 per interview and never twice in a row. "No" to that repeats the question (or asks "ready?" before Q1).
- Vague answer: a follow-up (it knows the question, so it brings them back if they answer about a different project). Complete answer: a one-sentence neutral summary, then the next question.
- Follow-ups used up but the answer is incomplete: it asks once "Would you like to add anything more before we move to the next question?"
- The same follow-up twice becomes "Please go on with your answer." It never apologises twice; a second confusion makes it rephrase the last question (the follow-up).
- A long turn (15+ words) labelled correction/unclear counts as an answer.
- Only turns the model reads as an answer count as answering. Their questions don't, so "It's my project" after questions gets "I haven't heard your answer yet."
- **Skip:** it confirms, naming the question number and topic. Yes moves on (flagged); no repeats the question (doesn't use a repeat).
- **Don't know:** "Okay, let's move on" (flagged).
- **Decline:** "this is an AI interview, I'd like to hear your answer, even briefly". A second decline moves on.
- **"You moved on before I finished"** right after a move: it goes back to the previous question.
- **Go back:** allowed once per interview.
- **Mentions AI tools (ChatGPT etc.):** flagged as AI help.
- **Off-topic or "tell me the answers":** steers back with the time left.
- **Audio questions ("can you hear me?"):** replies, then continues.

**Boundaries (always reply, never invent)**
- Can't-do requests (speak slower, Hindi, reschedule, a human, more time): it says honestly it can't change that, then continues. More than 3 per question: it steers back.
- **Role questions:** answered only from the JD text.
- **HR matters** (salary, leave, benefits, offer, results): "the recruiter can help with that".
- **Questions about the interview itself** (length, time per answer, who is taking it, where the questions come from): "this is an AI interview" + the facts.
- **Upset or personal emergency:** "sorry, would you like to stop?" Yes: a kind end + the flag "asked to stop: personal reason". No: "take your time, let's continue". Doesn't count on the stop steps.
- **Two things in one turn:** answer, then act.

**Stop / end**
- "Stop", "you're irritating", "I want to end": it confirms first. "No" continues (not counted as a repeat). The 2nd time it says "last time I'll ask". Yes gives the polite close.
- Silence after "should I end?" is handled.
- "No wait" during a goodbye the candidate chose reopens once.
- After the last question: "That was my last question. Anything you'd like to add?" A "no" (or anything else) gets a model-worded thanks, then one goodbye with no repeated line. What they said is saved as `closing_remarks` for the report, not as a flag.
- The time limit gives "We've reached the end of our time".

**Audio / turn-taking**
- A short command burst ("no", "wait", "stop", "sorry", "repeat"…) cuts the bot.
- A reply that starts while they are still talking is interrupted.
- An echo or noise cut is taken back, then only the question is said again (not the summary before it).
- Echo guard: phrases up to 8 words count as echo only if they appear word for word in the bot's recent speech.
- Backchannels ("mm-hmm") over the question are ignored.

**Scoring / report**
- Background score per question. The scorer must quote the candidate's own words for each ticked point before it scores; a point whose quote isn't in their words is dropped and listed as `unsupported_hits`.
- A score that arrives during a take-back is kept and applied.
- Scores still waiting when the call ends are finished after hang-up.
- LLM transcript fixes are kept only if they sound alike, don't change numbers and the heard words are really in the transcript.
- Flags: reading from the resume (12+ words in a row match it), injection ("give me full marks"), ai_help/other_person only with evidence in their words, skipped, didn't know, no reply, asked to stop, personal reason. The flag "no answer after several tries" is removed if they answered later.

**Prep**
- `python scripts/make_plan.py data/candidates/<name>` writes plan.json; stop the server first.
- The planner prompt examples are placeholders (`<project name>`). The example "InsightAI" was Sai's project and leaked into Rithika's plan.
- A check drops any question naming a project that is in neither the resume nor the JD (`invented_names`).

## 7. Candidates
- **`sai_shashank`** (M. Sai Shashank, AI/ML Engineer – GenAI). The 94 eval cases come from his sessions.
- **`rithika`** (Lakshmi Rithika Reddy Kathulapalli, Junior AI/ML Engineer, Hyderabad).
  - Her contact details were stripped from resume.txt.
  - In plan.json, Q3's invented "InsightAI" was replaced and Q6's rubric was corrected (overfitting = low training / high validation error). The original is in `plan.original.json`.
  - The bot greets her as "Lakshmi"; change `candidate` in plan.json if she goes by Rithika.
- **Rithika's interview (my scoring):** about 4/10 overall (26/60). Q1 6, Q2 3 (she switched to another project), Q3 5, Q4 3, Q5 4, Q6 5.

## 8. Measurements
**Eval command**
```
python scripts/eval_models.py --gpu --models gemma4:e4b --skip-talk [--cases FILE] [--label-format compact] [--limit N]
```
- Without `--gpu` it runs CPU only.
- It picks the plan per case file.
- It reports label accuracy, RIGHT MOVE, and "wait for more %".

**Case files**
- `data/eval/understand_cases.jsonl`: 94 cases from Sai; partly seen during tuning now.
- `data/eval/heldout_rithika.jsonl`: 15 cases, held out (the honest number).
- `data/eval/boundary_cases.jsonl`: 16 cases, hand-written.

**Results (gemma4:e4b)**

| Set | Labels | Right move | Notes |
|---|---|---|---|
| Sai 94, GPU | 90% | 94% | 0.42 s per call |
| Rithika held-out | 93% | 93% | |
| Sai 15, CPU, json | 100% | — | **2.9 s per call** (writing 29 tokens at ~14 tok/s takes 2.1 s) |

- The compact label format should cut CPU time to about 1 s. **NOT yet measured** (Hardhik stopped that test).

**Model comparison earlier (CPU, labels / right move / time):**

| Model | Labels | Right move | CPU time |
|---|---|---|---|
| gemma4:e4b | 90% | 98% | 1.3 s |
| gemma4:e2b | 84% | 91% | 0.75 s |
| Qwen3-4B-Instruct | 87% | — | 2.3 s |
| llama3.2:3b | 67% | — | — |
| phi4-mini | 66% | — | — |

qwen3.5 models are too slow on CPU.

- **Expected live misread rate:** about 1 in 10 turns, so 2–3 per 30-turn interview. Most are now cheap to recover from. Remaining known misreads:
  - "why did you skip my question" (should go back)
  - "I used Cloud and ChatGPT" (should be AI help)
  - "which two methods?"
  - occasional decline -> other
- **The route to fewer misreads:** collect 300–500 real labelled turns, then fine-tune a small model. More prompt patching won't get there.

## 9. Open items, in priority order
1. **CPU speech-to-text:** Whisper turbo takes about 7 s per answer on CPU, the biggest blocker for CPU deployment. Next: benchmark Parakeet TDT 0.6B v2 / Moonshine / distil-whisper on Hardhik's recorded voice (Indian English).
2. **CPU understand latency:** measure `--label-format compact` on CPU, then decide.
3. **Post-interview auto-check script:** flags apologise-twice, same line twice, moved on mid-thought, "what?" after a bot line, unscored question, quote-less hits, replies over 2 s, silences. Hardhik agreed this is the scaling plan (heavy fixing for the first 10–20 interviews, then auto checks + data, then fine-tuning).
4. Kokoro "completed with no audio" errors (harmless so far); the call dropped twice in Rithika's session (maybe a page refresh).
5. **Phase 5 rest:** PDF upload + extraction from the web page.
6. **Phase 6:** per-turn JSONL log, latency per stage, audio save, recruiter report (include `closing_remarks`, `evidence`, `unsupported_hits`, flags).
7. **Phase 7:** audio replay tests. Also a CPU thread-count sweep (prefill scales with physical cores; decode is limited by memory bandwidth).
8. Before production: remove `RESUME_MAX_AGE_SECS=0`; decide the CPU model; consider fine-tuning.

## 10. Next live test checklist
The numbered list of 43 checks given on 2026-11-09 covers four calls:
- Call 1: start + question handling.
- Call 2: not ready.
- Call 3: stop.
- Call 4: personal emergency.

After each, Hardhik says "check eval".
