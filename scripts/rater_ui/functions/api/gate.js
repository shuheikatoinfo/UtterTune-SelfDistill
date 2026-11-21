// Has this rater already taken the gate, and did they pass?
// The two-stage recruitment needs this: stage 2 is a separate contract, opened by invitation, and
// an invited worker may arrive in a different browser with no local record of having passed.
// The server's record is the one that counts.
export async function onRequestGet({ request, env }) {
  const rater = new URL(request.url).searchParams.get("rater") || "";
  const g = rater ? await env.RATER.get(`gate:${rater}`, "json") : null;
  return new Response(JSON.stringify(g ? { found: true, pass: g.pass, correct: g.correct, n: g.n }
                                       : { found: false }),
                      { headers: { "content-type": "application/json" } });
}
