const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

export interface ApiErrorItem {
  message?: string;
  code?: string;
  field?: string;
}

export interface ApiEnvelope<T> {
  ok: boolean;
  code: string;
  message: string;
  data: T;
  errors?: ApiErrorItem[];
}

export class ApiError extends Error {
  status: number;
  errors: ApiErrorItem[];

  constructor(message: string, status: number, errors: ApiErrorItem[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errors = errors;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  const payload = (await response.json()) as ApiEnvelope<T> | { detail?: string };
  if (!response.ok) {
    let message = "Request failed.";
    let errors: ApiErrorItem[] = [];
    if ("detail" in payload && !("ok" in payload)) {
      message = payload.detail ?? message;
    } else {
      const envelope = payload as ApiEnvelope<T>;
      message = envelope.message ?? envelope.errors?.[0]?.message ?? message;
      errors = envelope.errors ?? [];
    }
    throw new ApiError(message, response.status, errors);
  }

  if ("ok" in payload && payload.ok) {
    return payload.data;
  }

  throw new ApiError("Unexpected API response.", response.status);
}

export function getJson<T>(path: string) {
  return request<T>(path, { method: "GET" });
}

export function postJson<T>(path: string, body?: unknown) {
  return request<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body)
  });
}
