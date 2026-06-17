import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { api } from "./api";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown, init: Partial<{ ok: boolean; status: number; statusText: string }> = {}) {
  const ok = init.ok ?? true;
  return {
    ok,
    status: init.status ?? (ok ? 200 : 500),
    statusText: init.statusText ?? (ok ? "OK" : "Server Error"),
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

describe("api.health", () => {
  it("GETs /health and returns parsed JSON", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ status: "ok", uptime_seconds: 12 }));
    const res = await api.health();
    expect(res.status).toBe("ok");
    expect(res.uptime_seconds).toBe(12);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/health$/);
  });
});

describe("api.metrics", () => {
  it("requests JSON format", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ predictions_total: 5 }));
    await api.metrics();
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/metrics\?format=json$/);
  });
});

describe("api.transcript", () => {
  it("omits room_id when not given", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ entries: [] }));
    await api.transcript();
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/transcript$/);
  });

  it("URL-encodes the room id", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ entries: [] }));
    await api.transcript("a b/c");
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/transcript\?room_id=a%20b%2Fc$/);
  });
});

describe("api.feedback", () => {
  it("POSTs JSON body", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    await api.feedback({ category: "bug", message: "broken" });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ category: "bug", message: "broken" });
  });
});

describe("http error handling", () => {
  it("throws on non-2xx responses with status and body", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ error: "nope" }, { ok: false, status: 503, statusText: "Unavailable" }));
    await expect(api.health()).rejects.toThrow(/503/);
  });
});
