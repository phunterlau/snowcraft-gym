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

const viewport = requiredElement('replay-viewport');
const fileInput = requiredElement<HTMLInputElement>('recording-file');
const playButton = requiredElement<HTMLButtonElement>('play-toggle');
const scrubber = requiredElement<HTMLInputElement>('replay-scrubber');
const speedSelect = requiredElement<HTMLSelectElement>('replay-speed');
const status = requiredElement('replay-status');
const errorBox = requiredElement('replay-error');

let session: ReplaySession | null = null;

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  await openReplay(JSON.parse(await file.text()) as unknown, file.name);
});
playButton.addEventListener('click', () => session?.toggle());
scrubber.addEventListener('input', () => session?.seek(Number(scrubber.value)));
speedSelect.addEventListener('change', () => session?.setSpeed(Number(speedSelect.value)));

async function openReplay(value: unknown, label: string): Promise<void> {
  try {
    const recording = parseReplayRecording(value);
    session?.dispose();
    session = await ReplaySession.create(recording, label);
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

  dispose(): void {
    cancelAnimationFrame(this.animationFrame);
    this.playerRenderer.dispose();
    this.particleRenderer.dispose();
    this.arenaRenderer.dispose();
    this.renderer.dispose();
    this.assets.dispose();
    this.events.clear();
    viewport.replaceChildren();
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
  }
}

const query = new URLSearchParams(window.location.search);
const recordingUrl = query.get('recording');
if (recordingUrl) {
  try {
    const response = await fetch(recordingUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
