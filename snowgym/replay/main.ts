import '../../src/style.css';
import './style.css';
import { EventBus } from '../../src/core/EventBus';
import { AssetManager } from '../../src/engine/AssetManager';
import { Renderer } from '../../src/engine/Renderer';
import { ArenaRenderer } from '../../src/render/ArenaRenderer';
import { ParticleRenderer } from '../../src/render/ParticleRenderer';
import { PlayerRenderer } from '../../src/render/PlayerRenderer';
import { applyReplayTick, createReplayWorld } from './ReplayWorld';
import { parseReplayRecording, type ReplayRecording } from './ReplayRecording';
import { commanderOverlayAtTick } from './CommanderOverlay';
import {
  parseCommanderTrace,
  type CommanderTraceRecording,
} from '../orchestration/trace/CommanderTrace';

const viewport = requiredElement('replay-viewport');
const fileInput = requiredElement<HTMLInputElement>('recording-file');
const playButton = requiredElement<HTMLButtonElement>('play-toggle');
const scrubber = requiredElement<HTMLInputElement>('replay-scrubber');
const speedSelect = requiredElement<HTMLSelectElement>('replay-speed');
const status = requiredElement('replay-status');
const errorBox = requiredElement('replay-error');
const commanderOverlay = createCommanderOverlay();
const traceInput = createTraceInput();

let session: ReplaySession | null = null;
let pendingTrace: unknown = null;

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  try {
    pendingTrace = null;
    traceInput.value = '';
    commanderOverlay.hidden = true;
    await openReplay(JSON.parse(await file.text()) as unknown, file.name);
  } catch (error) {
    showError(message(error));
  }
});
traceInput.addEventListener('change', async () => {
  const file = traceInput.files?.[0];
  if (!file) return;
  try {
    pendingTrace = JSON.parse(await file.text()) as unknown;
    session?.loadCommanderTrace(pendingTrace);
  } catch (error) {
    showError(message(error));
  }
});
playButton.addEventListener('click', () => session?.toggle());
scrubber.addEventListener('input', () => session?.seek(Number(scrubber.value)));
speedSelect.addEventListener('change', () => session?.setSpeed(Number(speedSelect.value)));

async function openReplay(value: unknown, label: string): Promise<void> {
  try {
    const recording = parseReplayRecording(value);
    session?.dispose();
    session = await ReplaySession.create(recording, label);
    if (pendingTrace) session.loadCommanderTrace(pendingTrace);
    errorBox.hidden = true;
  } catch (error) {
    showError(message(error));
  }
}

class ReplaySession {
  private readonly assets = new AssetManager();
  private readonly renderer: Renderer;
  private readonly arenaRenderer: ArenaRenderer;
  private readonly playerRenderer: PlayerRenderer;
  private readonly particleRenderer: ParticleRenderer;
  private readonly world;
  private readonly events: EventBus;
  private animationFrame = 0;
  private lastTime = performance.now();
  private tick = 0;
  private speed = 1;
  private playing = true;
  private commanderTrace: CommanderTraceRecording | null = null;

  private constructor(
    private readonly recording: ReplayRecording,
    private readonly label: string,
  ) {
    const replayWorld = createReplayWorld(recording);
    this.world = replayWorld.world;
    this.events = replayWorld.events;
    this.renderer = new Renderer(viewport);
    this.renderer.frameArena(this.world.arena);
    this.arenaRenderer = new ArenaRenderer(this.renderer.scene, this.assets, this.world.arena);
    this.playerRenderer = new PlayerRenderer(this.renderer.scene, this.assets, this.world);
    this.particleRenderer = new ParticleRenderer(
      this.renderer.scene,
      this.assets,
      this.world,
      this.events,
    );
  }

  static async create(recording: ReplayRecording, label: string): Promise<ReplaySession> {
    const session = new ReplaySession(recording, label);
    await session.assets.loadAll();
    const finalTick = recording.outcome.finalTick;
    scrubber.min = '0';
    scrubber.max = String(finalTick);
    scrubber.step = '0.1';
    scrubber.value = '0';
    playButton.textContent = 'Pause';
    session.animationFrame = requestAnimationFrame(session.frame);
    return session;
  }

  toggle(): void {
    if (!this.playing && this.tick >= this.recording.outcome.finalTick) {
      this.tick = 0;
    }
    this.playing = !this.playing;
    this.lastTime = performance.now();
    playButton.textContent = this.playing ? 'Pause' : 'Play';
  }

  seek(tick: number): void {
    this.tick = Math.min(Math.max(tick, 0), this.recording.outcome.finalTick);
    this.playing = false;
    playButton.textContent = 'Play';
    this.render();
  }

  setSpeed(speed: number): void {
    if (Number.isFinite(speed) && speed > 0) this.speed = speed;
  }

  loadCommanderTrace(value: unknown): void {
    this.commanderTrace = parseCommanderTrace(value, this.recording);
    commanderOverlay.hidden = false;
    this.renderCommanderOverlay();
  }

  dispose(): void {
    cancelAnimationFrame(this.animationFrame);
    this.playerRenderer.dispose();
    this.particleRenderer.dispose();
    this.arenaRenderer.dispose();
    this.renderer.dispose();
    this.assets.dispose();
    this.events.clear();
    viewport.replaceChildren();
    commanderOverlay.hidden = true;
  }

  private readonly frame = (now: number): void => {
    const elapsed = Math.min((now - this.lastTime) / 1000, 0.25);
    this.lastTime = now;
    if (this.playing) {
      this.tick += elapsed * this.recording.simulationHz * this.speed;
      if (this.tick >= this.recording.outcome.finalTick) {
        this.tick = this.recording.outcome.finalTick;
        this.playing = false;
        playButton.textContent = 'Replay';
      }
    }
    this.render();
    this.animationFrame = requestAnimationFrame(this.frame);
  };

  private render(): void {
    applyReplayTick(this.world, this.recording, this.tick);
    this.playerRenderer.sync(0);
    this.particleRenderer.sync(0);
    this.renderer.render();
    scrubber.value = String(this.tick);

    const seconds = this.tick / this.recording.simulationHz;
    const outcome = this.recording.outcome;
    const ended = this.tick >= outcome.finalTick;
    const matchup = this.recording.configuration
      ? ` · ${this.recording.configuration.blueUnits}v${this.recording.configuration.redUnits}`
      : '';
    status.textContent = `${this.label} · seed ${this.recording.seed}${matchup} · tick ${Math.floor(this.tick)}/${outcome.finalTick} · ${seconds.toFixed(1)}s${ended ? ` · winner ${outcome.winner ?? 'none'}` : ''}`;
    this.renderCommanderOverlay();
  }

  private renderCommanderOverlay(): void {
    if (!this.commanderTrace) return;
    const state = commanderOverlayAtTick(this.commanderTrace, this.tick);
    const decision = state.plan.decision;
    const groups = decision.groups
      .map(({ role, order }) => `${role}: ${order.mission}/${order.approach}`)
      .join(' · ');
    const trajectory = state.trajectory
      ? state.trajectory.groups
          .map(
            ({ role, progress, stuckFraction }) =>
              `${role} ${progress} (stuck ${(stuckFraction * 100).toFixed(0)}%)`,
          )
          .join(' · ')
      : 'awaiting aggregate trajectory';
    const events = state.events
      .map(({ tick, label }) => `<li><span>t${tick}</span> ${escapeText(label)}</li>`)
      .join('');
    commanderOverlay.innerHTML = `
      <strong>Commander trace · plan v${state.plan.version}</strong>
      <div class="commander-intent">${escapeText(decision.intentSummary ?? 'No intent summary')}</div>
      <div>${escapeText(groups)}</div>
      <div>${escapeText(trajectory)}</div>
      <ol>${events}</ol>`;
  }
}

const query = new URLSearchParams(window.location.search);
const recordingUrl = query.get('recording');
const traceUrl = query.get('trace');
if (recordingUrl) {
  try {
    const response = await fetch(recordingUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (traceUrl) {
      const traceResponse = await fetch(traceUrl);
      if (!traceResponse.ok) throw new Error(`trace HTTP ${traceResponse.status}`);
      pendingTrace = (await traceResponse.json()) as unknown;
    }
    await openReplay((await response.json()) as unknown, recordingUrl);
  } catch (error) {
    showError(`Could not load ${recordingUrl}: ${message(error)}`);
  }
} else {
  status.textContent = 'Choose a SnowGym replay JSON file';
}

function requiredElement<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as T;
}

function showError(text: string): void {
  errorBox.textContent = text;
  errorBox.hidden = false;
  status.textContent = 'Replay unavailable';
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function createCommanderOverlay(): HTMLElement {
  const overlay = document.createElement('aside');
  overlay.id = 'commander-overlay';
  overlay.className = 'commander-overlay';
  overlay.setAttribute('aria-live', 'polite');
  overlay.hidden = true;
  document.getElementById('app')?.append(overlay);
  return overlay;
}

function createTraceInput(): HTMLInputElement {
  const label = document.createElement('label');
  label.className = 'file-button';
  label.textContent = 'Open commander trace';
  const input = document.createElement('input');
  input.id = 'commander-trace-file';
  input.type = 'file';
  input.accept = 'application/json,.json';
  label.append(input);
  document.querySelector('.replay-panel')?.prepend(label);
  return input;
}

function escapeText(value: string): string {
  const element = document.createElement('span');
  element.textContent = value;
  return element.innerHTML;
}
