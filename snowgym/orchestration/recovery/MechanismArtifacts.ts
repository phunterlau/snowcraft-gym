import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync, gunzipSync } from 'node:zlib';
import { benchmarkDigest } from '../benchmark/CommanderBackendBenchmark';
import { fileDigest } from '../benchmark/BackendBenchmarkArtifacts';
import {
  MECHANISM_ARMS,
  MECHANISM_CASES,
  MECHANISM_DELAYS,
  MECHANISM_VERSION,
} from './MechanismAudit';

const validName = (name: string) => /^[\w.-]+\.json(?:\.gz)?$/.test(name);
export function writeMechanismArtifact(directory: string, name: string, value: unknown): void {
  if (!validName(name)) throw new Error('invalid mechanism artifact name');
  const data = Buffer.from(`${JSON.stringify(value)}\n`);
  writeFileSync(join(directory, name), name.endsWith('.gz') ? gzipSync(data) : data, {
    flag: 'wx',
  });
}
export function readMechanismArtifact<T = unknown>(directory: string, name: string): T {
  if (!validName(name)) throw new Error('invalid mechanism artifact name');
  const data = readFileSync(join(directory, name));
  return JSON.parse((name.endsWith('.gz') ? gunzipSync(data) : data).toString('utf8'));
}
export function sealMechanismArtifacts(directory: string): void {
  const files = Object.fromEntries(
    readdirSync(directory)
      .sort()
      .map((name) => [name, fileDigest(join(directory, name))]),
  );
  const body = { format: MECHANISM_VERSION, files };
  writeMechanismArtifact(directory, 'manifest.json', { ...body, digest: benchmarkDigest(body) });
}
export function verifyMechanismArtifacts(directory: string): void {
  const { digest, ...body } = readMechanismArtifact<{
    digest: string;
    format: string;
    files: Record<string, string>;
  }>(directory, 'manifest.json');
  if (body.format !== MECHANISM_VERSION || benchmarkDigest(body) !== digest)
    throw new Error('mechanism manifest digest mismatch');
  const names = readdirSync(directory)
    .filter((name) => name !== 'manifest.json')
    .sort();
  if (benchmarkDigest(names) !== benchmarkDigest(Object.keys(body.files).sort()))
    throw new Error('mechanism artifact inventory mismatch');
  for (const name of names)
    if (!validName(name) || fileDigest(join(directory, name)) !== body.files[name])
      throw new Error(`mechanism artifact digest mismatch: ${name}`);
}

/** A sealed run of this exact declaration is recorded reuse, not a new seed allocation. */
export function verifyRegisteredMechanism(directory: string): string {
  verifyMechanismArtifacts(directory);
  const config = readMechanismArtifact<Record<string, unknown>>(directory, 'configuration.json');
  if (
    config.format !== MECHANISM_VERSION ||
    benchmarkDigest(config.cases) !== benchmarkDigest(MECHANISM_CASES) ||
    benchmarkDigest(config.arms) !== benchmarkDigest(MECHANISM_ARMS) ||
    benchmarkDigest(config.delays) !== benchmarkDigest(MECHANISM_DELAYS) ||
    config.horizon !== 300 ||
    config.scanHorizon !== 300 ||
    !['preflight', 'historical-only', 'full'].includes(String(config.mode))
  )
    throw new Error('registered mechanism declaration mismatch');
  return fileDigest(join(directory, 'manifest.json'));
}

/** Audits numeric seed metadata, including nested allocation ranges and schedule bounds. */
export function seedDeclarations(
  value: unknown,
  path = '',
  seedContext = false,
): { path: string; minimum: number; maximum: number }[] {
  if (seedContext && value === '__NONFINITE_SEED_AUDIT__')
    throw new Error(`non-finite seed declaration: ${path}`);
  if (Number.isSafeInteger(value) && seedContext)
    return [{ path, minimum: value as number, maximum: value as number }];
  if (Array.isArray(value)) {
    // Two-element seed declarations are conservatively treated as inclusive ranges.
    if (seedContext && value.length === 2 && value.every(Number.isSafeInteger))
      return [{ path, minimum: Math.min(...value), maximum: Math.max(...value) }];
    return value.flatMap((child, i) => seedDeclarations(child, `${path}[${i}]`, seedContext));
  }
  if (!value || typeof value !== 'object') return [];
  const object = value as Record<string, unknown>;
  const numeric = Object.values(object).filter((x): x is number => Number.isSafeInteger(x));
  if (
    seedContext &&
    Object.keys(object).some((key) => /min|max|start|end/i.test(key)) &&
    numeric.length >= 2
  )
    return [{ path, minimum: Math.min(...numeric), maximum: Math.max(...numeric) }];
  return Object.entries(object).flatMap(([key, child]) =>
    seedDeclarations(
      child,
      `${path}.${key}`,
      seedContext || /seed|partition|allocation/i.test(key),
    ),
  );
}
/** Python's historical JSON outputs may contain bare Infinity/NaN. Never rewrite strings or files. */
export function parseSeedJson(text: string): { value: unknown; nonFiniteValues: number } {
  let nonFiniteValues = 0;
  const normalized = text.replace(/"(?:[^"\\]|\\.)*"|-?\bInfinity\b|\bNaN\b/g, (token) => {
    if (token.startsWith('"')) return token;
    nonFiniteValues++;
    return '"__NONFINITE_SEED_AUDIT__"';
  });
  return { value: JSON.parse(normalized), nonFiniteValues };
}
export function auditSeedDocuments(
  documents: { path: string; value: unknown }[],
  minimum = 630000,
  maximum = 630119,
) {
  const declarations = documents.flatMap((document) =>
    seedDeclarations(document.value).map((row) => ({ ...row, file: document.path })),
  );
  const collisions = declarations.filter((row) => row.minimum <= maximum && row.maximum >= minimum);
  return {
    minimum,
    maximum,
    documents: documents.length,
    declarations: declarations.length,
    collisions,
  };
}

export interface MechanismRow {
  cohort: 'historical' | 'fresh';
  seed: number;
  roster: string;
  arm: string;
  delayDecisions: number;
  filename: string;
  firstChangedAction: number | null;
  interventionStatus: string;
  actualBindingChanges: number;
  metrics: {
    winner: string | null;
    damageDealt: number;
    damageReceived: number;
    blueAlive: number;
    rejectedActions: number;
    censored: boolean;
    decisions: number;
  };
}
export function pairedMechanismReport(rows: MechanismRow[], resamples = 10000, seed = 730001) {
  if (!Number.isSafeInteger(resamples) || resamples < 1 || !Number.isSafeInteger(seed))
    throw new Error('invalid bootstrap settings');
  let rng = seed >>> 0;
  const random = () => {
    rng = (Math.imul(1664525, rng) + 1013904223) >>> 0;
    return rng / 4294967296;
  };
  const results = [];
  for (const roster of ['5v5', '10v10', '6v10'])
    for (const delay of [0, 10, 20, 40, 80]) {
      const group = rows.filter(
        (row) => row.cohort === 'fresh' && row.roster === roster && row.delayDecisions === delay,
      );
      for (const [arm, control] of [
        ['refresh_binding', 'keep'],
        ['local_fire', 'keep'],
        ['reactivate', 'keep'],
        ['refresh_binding', 'reactivate'],
      ]) {
        const candidates = group.filter((row) => row.arm === arm).sort((a, b) => a.seed - b.seed);
        const controls = group.filter((row) => row.arm === control);
        if (
          new Set(candidates.map((row) => row.seed)).size !== candidates.length ||
          new Set(controls.map((row) => row.seed)).size !== controls.length ||
          controls.length !== candidates.length
        )
          throw new Error('duplicate or unmatched paired seeds');
        const differences = candidates.map((row) => {
          const paired = controls.find((other) => other.seed === row.seed);
          if (!paired) throw new Error('unmatched paired seed');
          return [
            Number(row.metrics.winner === 'blue') - Number(paired.metrics.winner === 'blue'),
            row.metrics.damageDealt -
              row.metrics.damageReceived -
              (paired.metrics.damageDealt - paired.metrics.damageReceived),
            row.metrics.blueAlive - paired.metrics.blueAlive,
          ];
        });
        const samples: number[][] = [[], [], []];
        if (differences.length)
          for (let i = 0; i < resamples; i++) {
            const sums = [0, 0, 0];
            for (let j = 0; j < differences.length; j++) {
              const selected = differences[Math.floor(random() * differences.length)];
              sums.forEach((_, k) => {
                sums[k] += selected[k];
              });
            }
            sums.forEach((sum, k) => samples[k].push(sum / differences.length));
          }
        results.push({
          roster,
          delayDecisions: delay,
          arm,
          control,
          pairs: differences.length,
          underCovered: differences.length < 20,
          metrics: Object.fromEntries(
            ['blueWinRateDifference', 'netDamageDifference', 'blueSurvivorDifference'].map(
              (name, k) => {
                const sorted = samples[k].sort((a, b) => a - b);
                return [
                  name,
                  differences.length
                    ? {
                        mean:
                          differences.reduce((sum, row) => sum + row[k], 0) / differences.length,
                        lower95: sorted[Math.floor((sorted.length - 1) * 0.025)],
                        upper95: sorted[Math.ceil((sorted.length - 1) * 0.975)],
                      }
                    : null,
                ];
              },
            ),
          ),
        });
      }
    }
  return {
    resamples,
    analysisSeed: seed,
    intervals: 'paired percentile bootstrap, exploratory, unadjusted',
    comparisons: results,
  };
}
