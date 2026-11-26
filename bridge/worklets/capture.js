// Capture worklet: the candidate's remote audio track is routed through here.
// It accumulates the 128-sample render quantums into 20 ms blocks (960 samples
// at 48 kHz), converts Float32 -> Int16, and posts each block to the main thread,
// which ships it to Python over the WebSocket. Python resamples 48k -> 16k.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._block = 960;            // 20 ms @ 48 kHz
    this._buf = new Float32Array(this._block);
    this._n = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];          // mono (first channel)
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this._buf[this._n++] = ch[i];
      if (this._n === this._block) {
        const pcm = new Int16Array(this._block);
        for (let j = 0; j < this._block; j++) {
          let s = Math.max(-1, Math.min(1, this._buf[j]));
          pcm[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
