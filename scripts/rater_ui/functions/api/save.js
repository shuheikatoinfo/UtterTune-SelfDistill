// One judgement, stored server-side. The browser's localStorage is still the rater's own safety
// net; this is the copy the analysis reads, so a cleared browser or a closed tab loses nothing.
// Same-origin, so the response is readable and a failure is visible to the page.
export async function onRequestPost({ request, env }) {
  try {
    const d = await request.json();
    if (!d || !d.rater || !d.blind_id) return json({ ok: false, error: "bad payload" }, 400);
    const key = `resp:${d.rater}:${d.kit || ""}|${d.blind_id}`;
    if (d.verdict === null || d.verdict === undefined) await env.RATER.delete(key);
    else await env.RATER.put(key, JSON.stringify({ ...d, at: Date.now() }));
    return json({ ok: true });
  } catch (e) {
    return json({ ok: false, error: String(e) }, 500);
  }
}
const json = (o, s = 200) =>
  new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json" } });
