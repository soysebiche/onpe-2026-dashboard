const ONPE_BASE = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend";

const FORWARD_HEADERS = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
  Referer: "https://resultadoelectoral.onpe.gob.pe/main/resumen",
  Origin: "https://resultadoelectoral.onpe.gob.pe",
  Accept: "application/json, text/plain, */*",
  "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
  "Sec-Fetch-Dest": "empty",
  "Sec-Fetch-Mode": "cors",
  "Sec-Fetch-Site": "same-origin",
  "X-Requested-With": "XMLHttpRequest",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Strip the worker path prefix and forward to ONPE
    const onpeUrl = ONPE_BASE + url.pathname + url.search;

    const upstream = await fetch(onpeUrl, {
      method: request.method,
      headers: FORWARD_HEADERS,
    });

    const contentType = upstream.headers.get("content-type") || "";

    // If ONPE returned HTML (blocked), return 502 so the caller knows
    if (!contentType.includes("application/json")) {
      return new Response(
        JSON.stringify({ error: "ONPE devolvio HTML - posible bloqueo upstream" }),
        { status: 502, headers: { "Content-Type": "application/json" } }
      );
    }

    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  },
};
