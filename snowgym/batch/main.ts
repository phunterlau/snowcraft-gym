import { createInterface } from 'node:readline';
import { BATCH_RESPONSE_FORMAT, BatchHost } from './BatchHost';

const host = new BatchHost();
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });

lines.on('line', (line) => {
  if (!line.trim()) return;
  try {
    const response = host.handle(JSON.parse(line) as unknown);
    process.stdout.write(`${JSON.stringify(response)}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(
      `${JSON.stringify({ format: BATCH_RESPONSE_FORMAT, requestId: null, error: 'invalid_request', message })}\n`,
    );
  }
});

lines.on('close', () => process.exit(0));
