/* eslint-disable @typescript-eslint/explicit-function-return-type */
/* global fetch, process */
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';

const args = process.argv.slice(2);
const server = option('--server') ?? 'http://127.0.0.1:8787';
const scripted = args.includes('--scripted');
const actionPath = option('--action');

if (scripted === Boolean(actionPath)) fail('choose exactly one of --scripted or --action PATH');

const snapshot = await request('GET', '/status');
const expectedStateHash = snapshot?.status?.stateHash;
if (typeof expectedStateHash !== 'string') fail('server status is missing stateHash');

const body = {
  expectedStateHash,
  idempotencyKey: `agent-${randomUUID()}`,
};
if (actionPath) body.action = JSON.parse(await readFile(actionPath, 'utf8'));

const result = await request('POST', scripted ? '/step-scripted' : '/step', body);
process.stdout.write(`${JSON.stringify(result)}\n`);

function option(name) {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  const value = args[index + 1];
  if (!value || value.startsWith('--')) fail(`${name} requires a value`);
  return value;
}

async function request(method, path, body) {
  const response = await fetch(`${server}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    fail(`server returned non-JSON HTTP ${response.status}`);
  }
  if (!response.ok) fail(`server returned HTTP ${response.status}: ${JSON.stringify(payload)}`);
  return payload;
}

function fail(message) {
  process.stderr.write(`strict-step: ${message}\n`);
  process.exit(1);
}
