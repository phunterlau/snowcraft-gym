/** Engine-independent actions accepted by the SnowGym control boundary. */
export type UnitAction =
  | { type: 'noop'; unitId: number }
  | { type: 'move'; unitId: number; x: number; y: number }
  | { type: 'throw'; unitId: number; x: number; y: number; power: number };

/** One decision for a whole team. Each unit may appear at most once. */
export interface TeamAction {
  actions: UnitAction[];
}
