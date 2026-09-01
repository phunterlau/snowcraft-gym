import { SIM } from '../../src/game/config';
import { Team } from '../../src/game/types';
import { RED_CONTROLLER_TYPES } from '../agents/opponents';
import { REPLAY_FORMAT } from '../replay/ReplayRecording';
import { MAX_TEAM_SIZE } from '../scenarios/Scenario';
import { getMapData, MAP_IDS, mapSpawns } from '../scenarios/maps';
import { SIMULATION_VERSION, STATE_HASH_VERSION, UPSTREAM_BASE_COMMIT } from './Version';

export const CAPABILITIES_FORMAT = 'snowgym.capabilities.v0' as const;

/** Machine-readable discovery document for clients and autonomous agents. */
export function snowGymCapabilities(): object {
  return {
    format: CAPABILITIES_FORMAT,
    apiVersion: 'snowgym.v0',
    simulationVersion: SIMULATION_VERSION,
    stateHashVersion: STATE_HASH_VERSION,
    replayFormat: REPLAY_FORMAT,
    upstreamBaseCommit: UPSTREAM_BASE_COMMIT,
    transport: {
      binding: 'loopback',
      episodeModel: 'single-shared',
      expectedStateHash: true,
      idempotencyKey: true,
      maxBodyBytes: 1_000_000,
    },
    endpoints: {
      health: { method: 'GET', path: '/health', mutates: false },
      capabilities: { method: 'GET', path: '/capabilities', mutates: false },
      status: { method: 'GET', path: '/status', mutates: false },
      reset: { method: 'POST', path: '/reset', mutates: true, guarded: true },
      step: { method: 'POST', path: '/step', mutates: true, requires: ['action'] },
      stepJoint: {
        method: 'POST',
        path: '/step-joint',
        mutates: true,
        requires: ['actions.blue', 'actions.red'],
      },
      stepScripted: { method: 'POST', path: '/step-scripted', mutates: true },
      autoplay: { method: 'POST', path: '/autoplay', mutates: true },
    },
    actions: {
      teamField: 'actions',
      types: {
        noop: { required: ['type', 'unitId'] },
        hold: { required: ['type', 'unitId'] },
        move: { required: ['type', 'unitId', 'x', 'y'] },
        throw: { required: ['type', 'unitId', 'x', 'y', 'power'] },
      },
      semantics: {
        omittedUnits: 'retain-current-movement-order',
        noop: 'does-not-cancel-current-movement-order',
        hold: 'cancels-current-movement-order',
        coordinates: 'world-space',
        powerRange: [0, 1],
      },
    },
    scenarios: {
      maxTeamSize: MAX_TEAM_SIZE,
      simulationHz: SIM.hz,
      decisionHz: Array.from({ length: SIM.hz }, (_, index) => index + 1).filter(
        (value) => SIM.hz % value === 0,
      ),
      redDifficulties: ['easy', 'normal', 'hard'],
      redControllers: RED_CONTROLLER_TYPES,
      maps: MAP_IDS.map((id) => ({
        id,
        name: getMapData(id).name,
        width: getMapData(id).width,
        height: getMapData(id).height,
        blueCapacity: mapSpawns(id, Team.Player).length,
        redCapacity: mapSpawns(id, Team.Enemy).length,
      })),
    },
    gymnasium: {
      environments: [
        { id: 'SnowGym/Squad-v0', maxTeamUnits: 3, configurable: false },
        { id: 'SnowGym/Squad-v1', maxTeamUnits: 8, configurable: true },
        { id: 'SnowGym/Squad-v2', maxTeamUnits: 10, configurable: true },
      ],
    },
    pettingZoo: {
      environment: {
        id: 'SnowGym/ParallelSquad-v0',
        api: 'parallel',
        agents: ['blue', 'red'],
        maxTeamUnits: MAX_TEAM_SIZE,
      },
      researchEnvironment: {
        id: 'SnowGym/ResearchParallelSquad-v0',
        visibilityRadiusUnits: 'world',
        latencyUnits: 'team-decisions',
        transforms: [
          'local-visibility',
          'action-delay',
          'observation-delay',
          'semantic-raster',
        ],
      },
    },
  };
}
