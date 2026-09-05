import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { BACKEND_BENCHMARK_VERSION, benchmarkDigest } from './CommanderBackendBenchmark';

export function writeBenchmarkJson(directory: string, filename: string, value: unknown): void {
  if (!/^[\w.-]+\.json$/.test(filename)) throw new Error('invalid artifact filename');
  writeFileSync(join(directory, filename), `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
}

export function fileDigest(path: string): string {
  return `sha256:${createHash('sha256').update(readFileSync(path)).digest('hex')}`;
}

export function sealBenchmarkArtifacts(directory: string): void {
  const files = Object.fromEntries(
    readdirSync(directory)
      .sort()
      .map((name) => [name, fileDigest(join(directory, name))]),
  );
  const body = { format: BACKEND_BENCHMARK_VERSION, files };
  writeBenchmarkJson(directory, 'manifest.json', {
    ...body,
    manifestDigest: benchmarkDigest(body),
  });
}

export function verifyBenchmarkArtifacts(directory: string): void {
  const manifest = JSON.parse(readFileSync(join(directory, 'manifest.json'), 'utf8'));
  const { manifestDigest, ...body } = manifest;
  if (manifest.format !== BACKEND_BENCHMARK_VERSION || benchmarkDigest(body) !== manifestDigest)
    throw new Error('manifest digest mismatch');
  if (
    benchmarkDigest(Object.keys(manifest.files).sort()) !==
    benchmarkDigest(
      readdirSync(directory)
        .filter((name) => name !== 'manifest.json')
        .sort(),
    )
  )
    throw new Error('artifact inventory mismatch');
  for (const [name, digest] of Object.entries(manifest.files)) {
    if (!/^[\w.-]+\.json$/.test(name) || fileDigest(join(directory, name)) !== digest)
      throw new Error(`artifact digest mismatch: ${name}`);
  }
}
