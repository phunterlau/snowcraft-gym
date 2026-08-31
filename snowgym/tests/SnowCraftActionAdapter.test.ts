import { describe, expect, it } from 'vitest';
import { EventBus } from '../../src/core/EventBus';
import { createEmptyArena } from '../../src/game/Arena';
import { PlayerState, Team } from '../../src/game/types';
import { World } from '../../src/game/World';
import { MovementSystem } from '../../src/systems/MovementSystem';
import { ThrowSystem } from '../../src/systems/ThrowSystem';
import { SnowCraftActionAdapter } from '../adapters/SnowCraftActionAdapter';

describe('SnowCraftActionAdapter', () => {
  it('applies per-unit movement without selection and enforces team ownership', () => {
    const world = new World(createEmptyArena(), 3);
    const blue = world.addPlayer(Team.Player, -3, 0);
    const red = world.addPlayer(Team.Enemy, 3, 0);
    const adapter = makeAdapter(world);

    const results = adapter.apply(Team.Player, {
      actions: [
        { type: 'move', unitId: blue.id, x: 1, y: 2 },
        { type: 'move', unitId: red.id, x: -1, y: 0 },
      ],
    });

    expect(blue.selected).toBe(false);
    expect(blue.moveTarget).toMatchObject({ x: 1, y: 2 });
    expect(results.map((result) => ({ accepted: result.accepted, reason: result.reason }))).toEqual(
      [
        { accepted: true, reason: undefined },
        { accepted: false, reason: 'wrong_team' },
      ],
    );
  });

  it('launches attributed projectiles and rejects duplicate actions', () => {
    const world = new World(createEmptyArena(), 3);
    const blue = world.addPlayer(Team.Player, -3, 0);
    const adapter = makeAdapter(world);

    const results = adapter.apply(Team.Player, {
      actions: [
        { type: 'throw', unitId: blue.id, x: 3, y: 0, power: 0.5 },
        { type: 'noop', unitId: blue.id },
      ],
    });

    expect(results[0].accepted).toBe(true);
    expect(results[1]).toMatchObject({ accepted: false, reason: 'duplicate_unit' });
    expect(world.snowballs).toHaveLength(1);
    expect(world.snowballs[0]).toMatchObject({ ownerId: blue.id, team: Team.Player });
  });

  it('distinguishes noop from an explicit hold order', () => {
    const world = new World(createEmptyArena(), 3);
    const blue = world.addPlayer(Team.Player, -3, 0);
    const adapter = makeAdapter(world);

    expect(
      adapter.apply(Team.Player, {
        actions: [{ type: 'move', unitId: blue.id, x: 3, y: 0 }],
      })[0].accepted,
    ).toBe(true);
    expect(blue.moveTarget).toMatchObject({ x: 3, y: 0 });

    expect(
      adapter.apply(Team.Player, {
        actions: [{ type: 'noop', unitId: blue.id }],
      })[0].accepted,
    ).toBe(true);
    expect(blue.moveTarget).toMatchObject({ x: 3, y: 0 });

    expect(
      adapter.apply(Team.Player, {
        actions: [{ type: 'hold', unitId: blue.id }],
      })[0].accepted,
    ).toBe(true);
    expect(blue.moveTarget).toBeNull();
    expect(blue.state).toBe(PlayerState.Idle);
  });

  it('rejects state-incompatible actions without partially mutating the unit', () => {
    const world = new World(createEmptyArena(), 3);
    const blue = world.addPlayer(Team.Player, -3, 0);
    const adapter = makeAdapter(world);
    blue.state = PlayerState.Throwing;

    const move = adapter.apply(Team.Player, {
      actions: [{ type: 'move', unitId: blue.id, x: 1, y: 2 }],
    });
    const throwing = adapter.apply(Team.Player, {
      actions: [{ type: 'throw', unitId: blue.id, x: 3, y: 0, power: 0.5 }],
    });

    expect(move[0]).toMatchObject({ accepted: false, reason: 'unavailable' });
    expect(throwing[0]).toMatchObject({ accepted: false, reason: 'unavailable' });
    expect(blue.moveTarget).toBeNull();
    expect(world.snowballs).toHaveLength(0);
  });
});

function makeAdapter(world: World): SnowCraftActionAdapter {
  const events = new EventBus();
  return new SnowCraftActionAdapter(
    world,
    new MovementSystem(world),
    new ThrowSystem(world, events),
  );
}
