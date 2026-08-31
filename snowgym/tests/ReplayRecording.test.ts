import { describe, expect, it } from 'vitest';
import { Team } from '../../src/game/types';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { SnowEnvironment } from '../core/SnowEnvironment';
import {
  parseReplayRecording,
  ReplayFormatError,
  type ReplayRecording,
} from '../replay/ReplayRecording';
import { applyReplayTick, createReplayWorld } from '../replay/ReplayWorld';

describe('SnowGym visual replay', () => {
  it('validates and interpolates detached observations into a render-only world', () => {
    const recording = makeRecording();
    expect(parseReplayRecording(structuredClone(recording))).toEqual(recording);

    const { world } = createReplayWorld(recording);
    const firstX = recording.frames[0].allies[0].x;
    const nextX = recording.frames[1].allies[0].x;
    applyReplayTick(world, recording, recording.ticksPerDecision / 2);

    expect(world.players).toHaveLength(6);
    expect(world.getPlayer(recording.frames[0].allies[0].id)?.position.x).toBeCloseTo(
      (firstX + nextX) / 2,
    );
    expect(world.time).toBeCloseTo(recording.ticksPerDecision / 2 / recording.simulationHz);
  });

  it('rejects malformed or mismatched visual recordings', () => {
    const recording = makeRecording() as unknown as Record<string, unknown>;
    recording.actions = [];
    expect(() => parseReplayRecording(recording)).toThrow(ReplayFormatError);
  });

  it('rejects malformed public-state hash sequences', () => {
    const recording = makeRecording() as unknown as Record<string, unknown>;
    recording.stateHashes = ['fnv1a64:0000000000000000'];

    expect(() => parseReplayRecording(recording)).toThrow(ReplayFormatError);
  });

  it('rejects a valid-looking hash that does not match its frame', () => {
    const recording = makeRecording() as unknown as Record<string, unknown>;
    const stateHashes = [...(recording.stateHashes as string[])];
    stateHashes[0] = 'fnv1a64:0000000000000000';
    recording.stateHashes = stateHashes;

    expect(() => parseReplayRecording(recording)).toThrow('does not match its frame');
  });
});

function makeRecording(): ReplayRecording {
  const environment = new SnowEnvironment();
  const policy = new SimpleBlueAgent();
  const initial = environment.reset(42);
  const initialStatus = environment.status();
  const action = policy.act(initial);
  const result = environment.step(action);
  const status = environment.status();

  return {
    format: 'snowgym.replay.v0',
    apiVersion: 'snowgym.v0',
    simulationVersion: status.simulationVersion,
    stateHashVersion: status.stateHashVersion,
    upstreamBaseCommit: status.upstreamBaseCommit,
    scenario: status.scenario,
    seed: status.seed,
    simulationHz: status.simulationHz,
    decisionHz: status.decisionHz,
    ticksPerDecision: status.ticksPerDecision,
    configuration: status.configuration,
    frames: [initial, environment.observe(Team.Player)],
    actions: [action],
    stateHashes: [initialStatus.stateHash, status.stateHash],
    outcome: {
      decisions: 1,
      terminated: status.terminated,
      truncated: status.truncated,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      finalTick: result.observation.tick,
    },
  };
}
