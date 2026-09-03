import { createHash } from 'node:crypto';
import type { AiDifficulty } from '../../../src/systems/AISystem';
import { runTrajectoryMockCommanderBattle } from '../examples/TrajectoryMockCommanderExample';

export const COMMANDER_LATENCY_BENCHMARK_FORMAT =
  'snowgym.commander-latency-benchmark.v0' as const;

export interface CommanderLatencyBenchmarkOptions {
  readonly seeds: readonly number[];
  readonly latencyTicks: readonly number[];
  readonly maxDecisions?: number;
  readonly blueUnits?: number;
  readonly redUnits?: number;
  readonly map?: string;
  readonly redDifficulty?: AiDifficulty;
}

export interface CommanderLatencyBenchmarkRow {
  readonly seed: number;
  readonly latencyTicks: number;
  readonly decisions: number;
  readonly finalTick: number;
  readonly winner: 'blue' | 'red' | 'draw' | null;
  readonly blueAlive: number;
  readonly redAlive: number;
  readonly commanderRequests: number;
  readonly rejectedActions: number;
  readonly plansActivated: number;
  readonly trajectorySignals: number;
  readonly responsesProcessed: number;
  readonly acceptedResponses: number;
  readonly repairedResponses: number;
  readonly rejectedResponses: number;
  readonly requestTimeouts: number;
  readonly requestFailures: number;
  readonly meanSourceAgeTicks: number | null;
}

export interface CommanderLatencyBenchmark {
  readonly format: typeof COMMANDER_LATENCY_BENCHMARK_FORMAT;
  readonly configuration: {
    readonly seeds: readonly number[];
    readonly latencyTicks: readonly number[];
    readonly maxDecisions: number;
    readonly blueUnits: number;
    readonly redUnits: number;
    readonly map: string;
    readonly redDifficulty: AiDifficulty;
  };
  readonly rows: readonly CommanderLatencyBenchmarkRow[];
  readonly summaries: readonly Record<string, number>[];
  readonly benchmarkDigest: string;
}

export async function runCommanderLatencyBenchmark(
  options: CommanderLatencyBenchmarkOptions,
): Promise<CommanderLatencyBenchmark> {
  const seeds = uniqueIntegers(options.seeds, 'seeds');
  const latencyTicks = uniqueIntegers(options.latencyTicks, 'latencyTicks', true);
  const configuration = {
    seeds,
    latencyTicks,
    maxDecisions: positive(options.maxDecisions ?? 400, 'maxDecisions'),
    blueUnits: positive(options.blueUnits ?? 10, 'blueUnits'),
    redUnits: positive(options.redUnits ?? 10, 'redUnits'),
    map: options.map ?? 'arena6.json',
    redDifficulty: options.redDifficulty ?? 'easy',
  } as const;
  const rows: CommanderLatencyBenchmarkRow[] = [];
  for (const latency of latencyTicks) {
    for (const seed of seeds) {
      const result = await runTrajectoryMockCommanderBattle({
        seed,
        latencyTicks: latency,
        maxDecisions: configuration.maxDecisions,
        blueUnits: configuration.blueUnits,
        redUnits: configuration.redUnits,
        map: configuration.map,
        redDifficulty: configuration.redDifficulty,
      });
      const processed = result.schedulerEvents.filter(
        (event) => event.type === 'response_processed',
      );
      rows.push({
        seed,
        latencyTicks: latency,
        decisions: result.decisions,
        finalTick: result.finalTick,
        winner: result.winner,
        blueAlive: result.blueAlive,
        redAlive: result.redAlive,
        commanderRequests: result.commanderRequests,
        rejectedActions: result.rejectedActions,
        plansActivated: result.planTraces.length,
        trajectorySignals: result.schedulerEvents.filter(
          ({ type }) => type === 'trajectory_signal',
        ).length,
        responsesProcessed: processed.length,
        acceptedResponses: processed.filter(({ status }) => status === 'accepted').length,
        repairedResponses: processed.filter(({ status }) => status === 'repaired').length,
        rejectedResponses: processed.filter(({ status }) => status === 'rejected').length,
        requestTimeouts: result.schedulerEvents.filter(
          ({ type }) => type === 'request_timed_out',
        ).length,
        requestFailures: result.schedulerEvents.filter(
          ({ type }) => type === 'request_failed',
        ).length,
        meanSourceAgeTicks: processed.length === 0 ? null : processed.reduce(
          (sum, event) => sum + event.sourceAgeTicks, 0,
        ) / processed.length,
      });
    }
  }
  const summaries = latencyTicks.map((latency) => summarizeLatency(
    latency, rows.filter((row) => row.latencyTicks === latency),
  ));
  const body = { format: COMMANDER_LATENCY_BENCHMARK_FORMAT, configuration, rows, summaries };
  return { ...body, benchmarkDigest: digest(body) };
}

export function auditCommanderLatencyBenchmark(value: CommanderLatencyBenchmark): void {
  const { benchmarkDigest, ...body } = value;
  if (value.format !== COMMANDER_LATENCY_BENCHMARK_FORMAT || digest(body) !== benchmarkDigest) {
    throw new Error('commander latency benchmark digest mismatch');
  }
  if (value.rows.length !== value.configuration.seeds.length * value.configuration.latencyTicks.length) {
    throw new Error('commander latency benchmark matrix is incomplete');
  }
}

function summarizeLatency(latencyTicks: number, rows: readonly CommanderLatencyBenchmarkRow[]) {
  const mean = (field: keyof CommanderLatencyBenchmarkRow) =>
    rows.reduce((sum, row) => sum + Number(row[field]), 0) / rows.length;
  return {
    latencyTicks,
    episodes: rows.length,
    blueWins: rows.filter(({ winner }) => winner === 'blue').length,
    redWins: rows.filter(({ winner }) => winner === 'red').length,
    draws: rows.filter(({ winner }) => winner === 'draw' || winner === null).length,
    meanDecisions: mean('decisions'),
    meanBlueAlive: mean('blueAlive'),
    meanRedAlive: mean('redAlive'),
    meanCommanderRequests: mean('commanderRequests'),
    meanResponsesProcessed: mean('responsesProcessed'),
    requestTimeouts: rows.reduce((sum, row) => sum + row.requestTimeouts, 0),
    requestFailures: rows.reduce((sum, row) => sum + row.requestFailures, 0),
    meanSourceAgeTicks: (() => {
      const ages = rows.flatMap(({ meanSourceAgeTicks }) =>
        meanSourceAgeTicks === null ? [] : [meanSourceAgeTicks]);
      return ages.length === 0 ? 0 : ages.reduce((sum, age) => sum + age, 0) / ages.length;
    })(),
    rejectedActions: rows.reduce((sum, row) => sum + row.rejectedActions, 0),
  };
}

function uniqueIntegers(values: readonly number[], name: string, allowZero = false): number[] {
  if (!Array.isArray(values) || values.length === 0 || values.some(
    (value) => !Number.isSafeInteger(value) || value < (allowZero ? 0 : 1),
  )) throw new RangeError(`${name} must contain ${allowZero ? 'non-negative' : 'positive'} integers`);
  if (new Set(values).size !== values.length) throw new RangeError(`${name} must be unique`);
  return [...values];
}

function positive(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be positive`);
  return value;
}

function digest(value: unknown): string {
  return `sha256:${createHash('sha256').update(canonical(value)).digest('hex')}`;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(
    (key) => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`,
  ).join(',')}}`;
  return JSON.stringify(value);
}
