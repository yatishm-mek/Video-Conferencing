"""Live orchestration: the Pipecat wiring, session objects, state machines and
turn-taking policy that drive one interview.

  pipeline.py      - builds and runs the Pipecat pipeline for a WebRTC session
  session.py       - the frame-processor sessions (classic + natural)
  controller.py    - the classic state machine
  natural / two_step / decide - the natural-engine turn logic
  turn_taking, echo_guard, echo, intents, timers, speech_clock, diagnostics
                     - the supporting turn / barge-in / timing helpers
"""
