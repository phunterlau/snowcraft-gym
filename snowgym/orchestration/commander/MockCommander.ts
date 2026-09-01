import type { CommanderClient, CommanderRequest, CommanderResponse } from './CommanderClient';

export type MockCommanderResponder = (
  request: CommanderRequest,
  callIndex: number,
) => CommanderResponse | Promise<CommanderResponse>;

export interface MockCommanderOptions {
  readonly latencyMs?: number;
  readonly sleep?: (milliseconds: number, signal?: AbortSignal) => Promise<void>;
}

/** Deterministic responder with injectable delay; production defaults to a real timer. */
export class MockCommander implements CommanderClient {
  private calls = 0;
  private readonly latencyMs: number;
  private readonly sleep: (milliseconds: number, signal?: AbortSignal) => Promise<void>;

  constructor(
    private readonly responder: MockCommanderResponder,
    options: MockCommanderOptions = {},
  ) {
    this.latencyMs = nonNegative(options.latencyMs ?? 1_500, 'latencyMs');
    this.sleep = options.sleep ?? abortableSleep;
  }

  async plan(request: CommanderRequest, signal?: AbortSignal): Promise<CommanderResponse> {
    const callIndex = this.calls++;
    await this.sleep(this.latencyMs, signal);
    return this.responder(structuredClone(request), callIndex);
  }
}

function abortableSleep(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, milliseconds);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(abortError());
      },
      { once: true },
    );
  });
}

function abortError(): Error {
  const error = new Error('commander request aborted');
  error.name = 'AbortError';
  return error;
}

function nonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be non-negative`);
  return value;
}
