// Playback worklet: renders the bot's TTS audio into the fake microphone.
// The main thread feeds it Int16 48 kHz PCM from Python (converted to Float32);
// process() drains a ring buffer into the output. On a 'flush' message it clears
// the buffer instantly (barge-in), so the candidate stops hearing the bot almost
// at once. The buffer is kept small so the only residual tail is the far-side
// jitter buffer.
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._cap = 48000 * 15;       // ~15 s ring buffer @ 48 kHz — must hold a whole
                                  // sentence even if Pipecat delivers it in a burst,
                                  // or the bot's voice drops off mid-sentence.
    this._ring = new Float32Array(this._cap);
    this._read = 0;
    this._write = 0;
    this._size = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d && d.cmd === 'flush') { this._read = this._write = this._size = 0; return; }
      // d is an Int16Array buffer of 48 kHz PCM.
      const pcm = new Int16Array(d);
      for (let i = 0; i < pcm.length; i++) {
        if (this._size >= this._cap) { this._read = (this._read + 1) % this._cap; this._size--; } // drop oldest
        this._ring[this._write] = pcm[i] / 0x8000;
        this._write = (this._write + 1) % this._cap;
        this._size++;
      }
    };
  }
  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    for (let i = 0; i < out.length; i++) {
      if (this._size > 0) {
        out[i] = this._ring[this._read];
        this._read = (this._read + 1) % this._cap;
        this._size--;
      } else {
        out[i] = 0;               // underrun -> silence
      }
    }
    return true;
  }
}
registerProcessor('playback-processor', PlaybackProcessor);
