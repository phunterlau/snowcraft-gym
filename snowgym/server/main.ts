import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { SnowGymService } from './SnowGymService';

const DEFAULT_PORT = 8787;
const MAX_BODY_BYTES = 1_000_000;
const service = new SnowGymService();
const port = parsePort(process.argv.slice(2));

const server = createServer(async (request, response) => {
  try {
    if (request.method === 'OPTIONS') {
      writeJson(response, 204, null);
      return;
    }

    const url = new URL(request.url ?? '/', 'http://127.0.0.1');
    const body = request.method === 'POST' ? await readJson(request) : undefined;
    const result = service.handle(request.method ?? 'GET', url.pathname, body);
    writeJson(response, result.status, result.body);
  } catch (error) {
    if (error instanceof HttpRequestError) {
      writeJson(response, 400, { error: error.code, message: error.message });
      return;
    }
    if (error instanceof SyntaxError) {
      writeJson(response, 400, { error: 'invalid_json', message: error.message });
      return;
    }
    console.error('SnowGym request failed', error);
    writeJson(response, 500, { error: 'internal_error', message: 'internal server error' });
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`SnowGym server listening on http://127.0.0.1:${port}`);
});

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => server.close(() => process.exit(0)));
}

function parsePort(args: string[]): number {
  const index = args.indexOf('--port');
  if (index === -1) return DEFAULT_PORT;
  const value = Number(args[index + 1]);
  if (!Number.isInteger(value) || value <= 0 || value > 65_535) {
    throw new RangeError('--port must be an integer between 1 and 65535');
  }
  return value;
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.length;
    if (bytes > MAX_BODY_BYTES) {
      throw new HttpRequestError('request_too_large', 'request body is too large');
    }
    chunks.push(buffer);
  }
  if (chunks.length === 0) return undefined;
  return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
}

class HttpRequestError extends Error {
  constructor(
    readonly code: 'request_too_large',
    message: string,
  ) {
    super(message);
  }
}

function writeJson(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, {
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'application/json; charset=utf-8',
  });
  response.end(body === null ? '' : JSON.stringify(body));
}
