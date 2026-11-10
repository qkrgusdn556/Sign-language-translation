import { hasMotionEvidence, trackedHandCount } from "./recognition.js?v=tcn-position-1";

export function createWindowDecoder({ holdMs = 300, threshold = 0.85, gap = 0.3 } = {}) {
  let history = [], emitted = null, previousEmitted = null;
  return {
    reset() { history = []; emitted = previousEmitted = null; },
    pause() { history = []; },
    reject(label) {
      if (emitted === label) emitted = previousEmitted;
      history = [];
    },
    update(result, now) {
      const ranked = [...result.scores].filter(item => typeof item.label === "string" && Number.isFinite(item.score) && item.score >= 0 && item.score <= 1).sort((a, b) => b.score - a.score);
      const best = ranked[0];
      if (!Number.isFinite(now)) return {status:"unknown",similarity:0,commit:false,reason:"low-score"};
      history = history.filter(item => now - item.at <= 600 && item.at < now);
      const strong = best && best.score >= threshold && best.score - (ranked[1]?.score || 0) >= gap;
      history.push({label:strong ? best.label : null,at:now});
      if (history.length > 4) history.shift();
      if (!strong) {
        return { status: "unknown", label: best?.label, similarity: best?.score || 0,
          reason: !best || best.score < threshold ? "low-score" : "small-margin", commit: false };
      }
      const votes = history.filter(item => item.label === best.label);
      const heldMs = now - votes[0].at;
      const commit = votes.length >= 3 && votes.length / history.length >= .75 && heldMs >= holdMs && emitted !== best.label;
      if (commit) { previousEmitted = emitted; emitted = best.label; }
      return { status: emitted === best.label ? "match" : "pending", label: best.label, similarity: best.score, commit,
        reason: commit ? "confirmed" : emitted === best.label ? "duplicate" : "holding",
        heldMs, votes: votes.length, observations: history.length };
    },
  };
}

export function createTcnStream({ infer, onResult, onError }) {
  const decoder = createWindowDecoder();
  let frames = [], lastSample = -Infinity, lastRequest = -Infinity, busy = false, generation = 0, missingSince = null, idleSince = null, inactiveSince = null;
  return {
    reset() { frames = []; lastSample = lastRequest = -Infinity; missingSince = idleSince = inactiveSince = null; generation++; decoder.reset(); },
    rejectCommit(label) { decoder.reject(label); },
    pause() { generation++;decoder.pause(); },
    missing(now) {
      if (missingSince === null) { missingSince = now; generation++; decoder.pause(); }
      onResult({status:"unknown",similarity:0,reason:"tracking-missing",commit:false});
      if (now - missingSince > 220) this.reset();
    },
    push(vector, now, { active = true, moving = active } = {}) {
      if (vector.length !== 130 || !vector.every(Number.isFinite) || !trackedHandCount(vector)) return;
      if (!active) inactiveSince ??= now;
      else inactiveSince = null;
      // The activity grace period preserves continuity, not permission to confirm at rest.
      if (!active || !moving) {
        if (idleSince === null) { idleSince = now; generation++; decoder.pause(); }
        onResult({status:"unknown",similarity:0,reason:active?"motion-settled":"inactive-input",commit:false});
      }
      if (missingSince !== null) {
        if (now - missingSince > 220) this.reset();
        missingSince = null;
        lastSample = now - 75;
      }
      if (now - lastSample < 75) return;
      if (now - lastSample > 300) { frames = []; generation++; decoder.reset(); }
      lastSample = now;
      frames.push([...vector]);
      if (frames.length > 20) frames.shift();
      // Keep pre-roll through quiet moments; never run or confirm while inactive.
      if (!active || !moving) {
        if (inactiveSince !== null && now - inactiveSince >= 1000) decoder.reset();
        return;
      }
      idleSince = null;
      if (frames.length >= 12 && !hasMotionEvidence(frames)) {
        generation++;
        decoder.pause();
        onResult({ status: "unknown", similarity: 0, reason: "insufficient-motion", commit: false });
        return;
      }
      if (frames.length < 12 || busy || now - lastRequest < 150) return;
      busy = true;
      lastRequest = now;
      const requestGeneration = generation;
      const snapshot = frames.map(frame => [...frame]);
      Promise.resolve().then(() => infer(snapshot)).then(result => {
        if (generation === requestGeneration) onResult(decoder.update(result, now));
      }).catch(error => {
        if (generation === requestGeneration) onError(error);
      }).finally(() => { busy = false; });
    },
  };
}
