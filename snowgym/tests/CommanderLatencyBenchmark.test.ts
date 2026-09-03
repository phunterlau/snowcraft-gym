import { describe, expect, it } from 'vitest';
import {
  auditCommanderLatencyBenchmark,
  runCommanderLatencyBenchmark,
} from '../orchestration/benchmark/CommanderLatencyBenchmark';

describe('commander latency benchmark', () => {
  it('runs a complete deterministic seed-by-latency matrix with an audited digest', async () => {
    const options = { seeds: [3, 4], latencyTicks: [0, 30], maxDecisions: 30 };
    const first = await runCommanderLatencyBenchmark(options);
    const second = await runCommanderLatencyBenchmark(options);
    expect(second).toEqual(first);
    expect(first.rows).toHaveLength(4);
    expect(first.summaries).toHaveLength(2);
    expect(first.rows.every(({ rejectedActions }) => rejectedActions === 0)).toBe(true);
    expect(first.rows.every(({ requestFailures }) => requestFailures === 0)).toBe(true);
    expect(() => auditCommanderLatencyBenchmark(first)).not.toThrow();
    const tampered = structuredClone(first);
    (tampered.rows[0] as { blueAlive: number }).blueAlive++;
    expect(() => auditCommanderLatencyBenchmark(tampered)).toThrow('digest mismatch');
  });

  it('rejects duplicate axes before starting a battle', async () => {
    await expect(runCommanderLatencyBenchmark({ seeds: [1, 1], latencyTicks: [0] }))
      .rejects.toThrow('seeds must be unique');
  });
});
