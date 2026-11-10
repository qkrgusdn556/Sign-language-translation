const JAMO = /^[ㄱ-ㅎㅏ-ㅣ]$/u;

export function createFingerspelling(assemble, holdMs = 300) {
  let letters = [], candidate = null, since = 0, latched = null;
  return {
    text: () => assemble(letters),
    letters: () => [...letters],
    resetCandidate() { candidate = null; since = 0; },
    release() { candidate = null; since = 0; latched = null; },
    clear() { letters = []; this.release(); },
    backspace() { letters.pop(); this.release(); },
    observe(label, confidence, now, {threshold = .8} = {}) {
      if (!JAMO.test(label || "") || !Number.isFinite(confidence) || confidence < threshold) {
        this.resetCandidate();
        return false;
      }
      if (label !== candidate) { candidate = label; since = now; return false; }
      if (now - since < holdMs || latched === label) return false;
      letters.push(label);
      latched = label;
      return true;
    },
  };
}

export function createAutomaticFingerspelling(assemble, { onText, onEnd, threshold = 0.9,
  holdMs = 450, releaseMs = 400, joinMs = 1400, candidateTtl = 500 } = {}) {
  const spelling = createFingerspelling(assemble, holdMs);
  let lastCandidate = -Infinity, lastObservation = -Infinity, otherSince = null;
  let motionAt = -Infinity, armed = false;
  return {
    noteMotion(now) { motionAt = now; armed = true; },
    status(now) {
      if(!armed) return spelling.text() ? "already-entered" : "motion-required";
      return now-motionAt>2500 ? "motion-expired" : "holding";
    },
    text: () => spelling.text(),
    blocksWord(now) { return now - lastCandidate <= candidateTtl; },
    reset() { spelling.clear(); lastCandidate = lastObservation = motionAt = -Infinity; otherSince = null; armed = false; },
    finish() {
      const text = spelling.text();
      spelling.clear(); lastCandidate = lastObservation = -Infinity; otherSince = null;
      if (text) onEnd?.(text);
    },
    pause({ lost = false } = {}) { spelling.resetCandidate(); otherSince = null; if (lost) armed = false; },
    tick(now) { if (spelling.text() && now - lastCandidate > joinMs) this.finish(); },
    observe(label, confidence, now, {threshold: observationThreshold=threshold, requireMotion=true}={}) {
      if (!Number.isFinite(now)) return false;
      this.tick(now);
      if (now - lastObservation > candidateTtl) spelling.resetCandidate();
      lastObservation = now;
      if (JAMO.test(label || "") && Number.isFinite(confidence) && confidence >= observationThreshold) {
        if (requireMotion && (!armed || now - motionAt > 2500)) { spelling.resetCandidate(); return false; }
        lastCandidate = now;
        otherSince = null;
        const accepted = spelling.observe(label, confidence, now, {threshold: observationThreshold});
        if (accepted) { armed = false; onText?.(spelling.text()); }
        return accepted;
      }
      spelling.resetCandidate();
      if (label === "기타" && confidence >= observationThreshold) {
        if (otherSince === null) otherSince = now;
        if (now - otherSince >= releaseMs) spelling.release();
      } else otherSince = null;
      return false;
    },
  };
}
