import type { TeamAction } from '../actions/UnitAction';
import type { Observation } from '../observations/Observation';

/**
 * Common policy boundary for scripted, learned, and remote controllers.
 *
 * `act` is called once per simulation tick with the fixed timestep `dt`; a
 * policy that only decides at a lower rate keeps its own cadence internally.
 * Controllers must never hold engine entities or mutate world state beyond
 * the returned semantic actions.
 */
export interface TeamController {
  act(observation: Observation, dt: number): TeamAction;
}
