// Use image-space XY, not normalized finger depth or the shoulder reference.
export function createInputActivity({profile = 'sign'} = {}) {
  const fingerspell = profile === 'fingerspell';
  let tracks = [], lastSample = -Infinity, lastMotion = -Infinity, missingSince = null, observedMoving = false;
  function delta(a, b) {
    const cost = (swapped) => a.reduce((sum, hand, h) => sum + hand.reduce((n, p, i) => {
      const q = b[swapped ? 1 - h : h][i];
      return n + Math.hypot(p.x - q.x, p.y - q.y) / hand.length;
    }, 0) / a.length, 0);
    return Math.min(cost(false), a.length === 2 ? cost(true) : Infinity);
  }
  function average(group) {
    return group[0].map((hand, h) => hand.map((_, i) => ({
      x: group.reduce((n, f) => n + f[h][i].x, 0) / group.length,
      y: group.reduce((n, f) => n + f[h][i].y, 0) / group.length,
    })));
  }
  function shapeDelta(a,b) {
    const scale=hand=>Math.max(.02,Math.hypot(hand[9].x-hand[0].x,hand[9].y-hand[0].y));
    const first=a[0],second=b[0],sa=scale(first),sb=scale(second);
    return [4,8,12,16,20].reduce((sum,i)=>sum+Math.hypot(
      (first[i].x-first[0].x)/sa-(second[i].x-second[0].x)/sb,
      (first[i].y-first[0].y)/sa-(second[i].y-second[0].y)/sb),0)/5;
  }
  return {
    reset() { tracks = []; lastSample = lastMotion = -Infinity; missingSince = null; observedMoving = false; },
    update(hands, now) {
      if (!hands.length || hands.some(hand => hand.length !== 21 || hand.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y)))) {
        if (missingSince === null) missingSince = now;
        if (now - missingSince > 220) this.reset();
        tracks = [];
        observedMoving = false;
        return { moving: false, active: false };
      }
      if (missingSince !== null) {
        if (now - missingSince > 220) this.reset();
        lastSample = now - 75;
        missingSince = null;
      }
      if (now - lastSample < 75) return { moving: observedMoving, active: now - lastMotion <= 450 };
      if (now - lastSample > 300) this.reset();
      lastSample = now;
      const available = tracks.filter(track=>now-track.at<=220);
      const observed = [];
      let moving = false;
      for(const hand of hands.slice(0,2)) {
        available.sort((a,b)=>Math.hypot(a.hand[0].x-hand[0].x,a.hand[0].y-hand[0].y)-Math.hypot(b.hand[0].x-hand[0].x,b.hand[0].y-hand[0].y));
        let track=available.shift();
        if(!track || Math.hypot(track.hand[0].x-hand[0].x,track.hand[0].y-hand[0].y)>.25) track={frames:[]};
        track.hand=hand.map(p=>({x:p.x,y:p.y}));track.at=now;
        track.frames.push([track.hand]);
        if(track.frames.length>(fingerspell ? 8 : 12))track.frames.shift();
        observed.push(track);
        const smooth=[];
        for(let i=2;i<track.frames.length;i++)smooth.push(average(track.frames.slice(i-2,i+1)));
        if (smooth.length < (fingerspell ? 3 : 4)) continue;
        // Words retain the original motion gate; jamo use a shorter, smaller gesture.
        const continuing = now - lastMotion <= 450;
        const stepThreshold = fingerspell ? .0025 : continuing ? .002 : .004;
        const excursionThreshold = fingerspell ? .012 : continuing ? .02 : .035;
        let steps = 0, excursion = 0, shapeSteps=0, shapeExcursion=0;
        const shapeStep=fingerspell ? .04 : continuing ? .03 : .05, shapeExtent=fingerspell ? .2 : continuing ? .2 : .35;
        for (let i = 1; i < smooth.length; i++) {
          if (delta(smooth[i - 1], smooth[i]) >= stepThreshold) steps++;
          excursion = Math.max(excursion, delta(smooth[0], smooth[i]));
          if(shapeDelta(smooth[i-1],smooth[i])>=shapeStep)shapeSteps++;
          shapeExcursion=Math.max(shapeExcursion,shapeDelta(smooth[0],smooth[i]));
        }
        const requiredSteps = fingerspell ? 2 : 3;
        moving ||= steps >= requiredSteps && excursion >= excursionThreshold && delta(smooth.at(-2), smooth.at(-1)) >= stepThreshold;
        moving ||= shapeSteps>=requiredSteps && shapeExcursion>=shapeExtent && shapeDelta(smooth.at(-2),smooth.at(-1))>=shapeStep;
      }
      // Losing the other hand must not erase the surviving hand's motion history.
      tracks=[...observed,...available].slice(0,2);
      if (moving) lastMotion = now;
      observedMoving = moving;
      return { moving, active: now - lastMotion <= 450 };
    },
  };
}
