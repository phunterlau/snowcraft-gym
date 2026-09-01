import { SnowGymService, type ServiceResponse } from '../server/SnowGymService';
import { snowGymCapabilities } from '../protocol/Capabilities';

export const BATCH_PROTOCOL_VERSION = 'snowgym.batch.v0' as const;
export const BATCH_REQUEST_FORMAT = 'snowgym.batch-request.v0' as const;
export const BATCH_RESPONSE_FORMAT = 'snowgym.batch-response.v0' as const;

type BatchOperation = 'handshake' | 'status' | 'reset' | 'step' | 'stepJoint' | 'close';

interface BatchItem {
  worldId: string;
  body?: unknown;
}

interface BatchRequest {
  format: typeof BATCH_REQUEST_FORMAT;
  requestId: string;
  operation: BatchOperation;
  items: BatchItem[];
}

export interface BatchItemResponse {
  worldId: string;
  status: number;
  body: unknown;
}

export interface BatchResponse {
  format: typeof BATCH_RESPONSE_FORMAT;
  requestId: string;
  operation: BatchOperation;
  results: BatchItemResponse[];
}

/** Persistent, transport-independent owner of isolated SnowGym services. */
export class BatchHost {
  private readonly worlds = new Map<string, SnowGymService>();

  handle(value: unknown): BatchResponse {
    const request = parseBatchRequest(value);
    if (request.operation === 'handshake') {
      return {
        format: BATCH_RESPONSE_FORMAT,
        requestId: request.requestId,
        operation: request.operation,
        results: [
          {
            worldId: 'host',
            status: 200,
            body: {
              protocolVersion: BATCH_PROTOCOL_VERSION,
              capabilities: snowGymCapabilities(),
              operations: ['status', 'reset', 'step', 'stepJoint', 'close'],
              isolation: 'per-world-explicit-result',
            },
          },
        ],
      };
    }
    const operation = request.operation as Exclude<BatchOperation, 'handshake'>;
    const results = request.items.map((item) => this.handleItem(operation, item));
    return {
      format: BATCH_RESPONSE_FORMAT,
      requestId: request.requestId,
      operation: request.operation,
      results,
    };
  }

  get size(): number {
    return this.worlds.size;
  }

  private handleItem(
    operation: Exclude<BatchOperation, 'handshake'>,
    item: BatchItem,
  ): BatchItemResponse {
    if (operation === 'close') {
      const existed = this.worlds.delete(item.worldId);
      return { worldId: item.worldId, status: 200, body: { closed: existed } };
    }
    let service = this.worlds.get(item.worldId);
    let created = false;
    if (operation === 'reset' && service === undefined) {
      service = new SnowGymService();
      this.worlds.set(item.worldId, service);
      created = true;
    }
    if (service === undefined) {
      return {
        worldId: item.worldId,
        status: 404,
        body: { error: 'world_not_found' },
      };
    }
    const route = batchRoute(operation);
    const response: ServiceResponse = service.handle(route.method, route.path, item.body);
    if (created && response.status !== 200) this.worlds.delete(item.worldId);
    return { worldId: item.worldId, ...response };
  }
}

function batchRoute(operation: Exclude<BatchOperation, 'handshake' | 'close'>): {
  method: string;
  path: string;
} {
  if (operation === 'status') return { method: 'GET', path: '/status' };
  if (operation === 'reset') return { method: 'POST', path: '/reset' };
  if (operation === 'step') return { method: 'POST', path: '/step' };
  return { method: 'POST', path: '/step-joint' };
}

function parseBatchRequest(value: unknown): BatchRequest {
  if (!isRecord(value)) throw new Error('batch request must be an object');
  const allowed = new Set(['format', 'requestId', 'operation', 'items']);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`unknown batch request fields: ${unknown.sort().join(', ')}`);
  if (value.format !== BATCH_REQUEST_FORMAT) {
    throw new Error(`batch request format must be ${BATCH_REQUEST_FORMAT}`);
  }
  if (typeof value.requestId !== 'string' || !/^[A-Za-z0-9._:-]{1,128}$/.test(value.requestId)) {
    throw new Error('batch requestId is invalid');
  }
  const operations = new Set<BatchOperation>([
    'handshake',
    'status',
    'reset',
    'step',
    'stepJoint',
    'close',
  ]);
  if (typeof value.operation !== 'string' || !operations.has(value.operation as BatchOperation)) {
    throw new Error('batch operation is invalid');
  }
  if (!Array.isArray(value.items)) throw new Error('batch items must be an array');
  if (value.operation === 'handshake' && value.items.length !== 0) {
    throw new Error('handshake items must be empty');
  }
  if (value.operation !== 'handshake' && value.items.length === 0) {
    throw new Error('batch operation requires at least one item');
  }
  const ids = new Set<string>();
  const items = value.items.map((item, index) => {
    if (!isRecord(item)) throw new Error(`batch item ${index} must be an object`);
    const itemUnknown = Object.keys(item).filter((key) => !['worldId', 'body'].includes(key));
    if (itemUnknown.length) throw new Error(`batch item ${index} has unknown fields`);
    if (typeof item.worldId !== 'string' || !/^[A-Za-z0-9._:-]{1,64}$/.test(item.worldId)) {
      throw new Error(`batch item ${index} worldId is invalid`);
    }
    if (ids.has(item.worldId)) throw new Error(`duplicate batch worldId ${item.worldId}`);
    ids.add(item.worldId);
    return { worldId: item.worldId, ...(item.body === undefined ? {} : { body: item.body }) };
  });
  return {
    format: BATCH_REQUEST_FORMAT,
    requestId: value.requestId,
    operation: value.operation as BatchOperation,
    items,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
