import type { TeamAction } from '../actions/UnitAction';
import type { Observation } from '../observations/Observation';

/** Common policy boundary for scripted, learned, and remote controllers. */
export interface TeamController {
  act(observation: Observation): TeamAction;
}
