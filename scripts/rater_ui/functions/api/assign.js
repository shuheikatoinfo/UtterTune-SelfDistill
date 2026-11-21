// Hand out the emptiest unit (a block, or a naturalness group) instead of hashing the rater id.
//
// Hashing spreads raters at random, and random is not even: 24 raters over 4 groups leave the
// thinnest group at 4 about as often as at 6, and reach a 6/6/6/6 split roughly 1% of the time.
// The consequence is real but quiet -- some stimuli get eight ratings and others four, so the
// thinnest cell's interval is the widest and part of the fee buys nothing.
//
// The assignment is stored per rater, so a reload, a different device or a resumed session all
// return the same unit.
//
// KV listing is eventually consistent, so a burst of arrivals can each see a tally that does not
// yet include the others -- twenty-four raters assigned in one second all read an empty list and
// would all take unit 0. Ties are therefore broken by a hash of the rater id, not by index: when
// the tally has not caught up every unit ties and the choice is random, which is where hashing
// started; as it catches up the minimum is real and the thin unit wins. The scheme degrades to
// random and converges to balanced, instead of collapsing onto one unit.
export async function onRequestGet({ request, env }) {
  const u = new URL(request.url);
  const study = (u.searchParams.get("study") || "").replace(/[^a-z0-9_-]/gi, "");
  const rater = (u.searchParams.get("rater") || "").replace(/[^a-z0-9_-]/gi, "");
  const n = Math.max(1, Math.min(50, parseInt(u.searchParams.get("n") || "0", 10) || 0));
  if (!study || !rater || !n) return json({ error: "study, rater and n are required" }, 400);

  const mine = await env.RATER.get(`assign:${study}:${rater}`);
  if (mine !== null) return json({ unit: mine, fresh: false });

  const counts = new Array(n).fill(0);
  let cursor;
  do {
    const page = await env.RATER.list({ prefix: `assign:${study}:`, cursor });
    for (const k of page.keys) {
      const v = await env.RATER.get(k.name);
      const i = parseInt(v, 10);
      if (Number.isInteger(i) && i >= 0 && i < n) counts[i]++;
    }
    cursor = page.list_complete ? null : page.cursor;
  } while (cursor);

  const lo = Math.min(...counts);
  const tied = [];
  for (let i = 0; i < n; i++) if (counts[i] === lo) tied.push(i);
  let h = 2166136261 >>> 0;
  for (const ch of rater) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  const unit = tied[h % tied.length];
  await env.RATER.put(`assign:${study}:${rater}`, String(unit));
  return json({ unit: String(unit), fresh: true, counts });
}
const json = (o, s = 200) =>
  new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json" } });
