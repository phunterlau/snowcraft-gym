/* global process, console, setTimeout, fetch, AbortSignal, URL, document, HTMLElement, HTMLCanvasElement, HTMLInputElement, Event */
/* eslint-disable @typescript-eslint/explicit-function-return-type */
// Browser acceptance check for the versioned SnowGym visual replay viewer.
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const url =
  process.env.REPLAY_URL ||
  'http://127.0.0.1:5173/replay.html?recording=/replays/blue-seed-42.json';
const screenshotPath = process.argv[2] || '/tmp/snowgym-replay.png';
const executablePath =
  process.env.CHROME_PATH ||
  (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    : 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe');

const ownedServer = await ensureReplayServer(url);
const expected = await loadExpectedReplay(url);
const errors = [];
let browser;
try {
  browser = await puppeteer.launch({
    executablePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--window-size=1280,800',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('requestfailed', (request) =>
    errors.push(`requestfailed: ${request.url()} ${request.failure()?.errorText ?? ''}`),
  );

  await page.goto(url, { waitUntil: 'networkidle0', timeout: 20_000 });
  try {
    await page.waitForSelector('#replay-viewport canvas', { timeout: 10_000 });
  } catch (error) {
    const diagnostic = await page.evaluate(() => ({
      body: document.body.innerText,
      html: document.body.innerHTML.slice(0, 2_000),
    }));
    console.log('DIAGNOSTIC', JSON.stringify(diagnostic));
    console.log('ERRORS', JSON.stringify(errors, null, 2));
    await page.screenshot({ path: screenshotPath });
    throw error;
  }
  await new Promise((resolve) => setTimeout(resolve, 700));

  const initial = await page.evaluate(() => {
    const canvas = document.querySelector('#replay-viewport canvas');
    const status = document.querySelector('#replay-status');
    const play = document.querySelector('#play-toggle');
    const scrubber = document.querySelector('#replay-scrubber');
    const commander = document.querySelector('#commander-overlay');
    return {
      hasCanvas: canvas instanceof HTMLCanvasElement,
      canvasWidth: canvas instanceof HTMLCanvasElement ? canvas.width : 0,
      canvasHeight: canvas instanceof HTMLCanvasElement ? canvas.height : 0,
      hasWebGl: canvas instanceof HTMLCanvasElement && Boolean(canvas.getContext('webgl2')),
      status: status?.textContent ?? '',
      playLabel: play?.textContent ?? '',
      maxTick: scrubber instanceof HTMLInputElement ? Number(scrubber.max) : 0,
      commanderVisible: commander instanceof HTMLElement && !commander.hidden,
      commanderText: commander?.textContent ?? '',
    };
  });

  await page.evaluate(() => {
    const scrubber = document.querySelector('#replay-scrubber');
    if (!(scrubber instanceof HTMLInputElement)) throw new Error('missing replay scrubber');
    scrubber.value = scrubber.max;
    scrubber.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await new Promise((resolve) => setTimeout(resolve, 100));

  const finalState = await page.evaluate(() => ({
    status: document.querySelector('#replay-status')?.textContent ?? '',
    playLabel: document.querySelector('#play-toggle')?.textContent ?? '',
    error: document.querySelector('#replay-error')?.textContent ?? '',
    commanderText: document.querySelector('#commander-overlay')?.textContent ?? '',
  }));

  await page.click('#play-toggle');
  await new Promise((resolve) => setTimeout(resolve, 150));
  const replayedState = await page.evaluate(() => ({
    status: document.querySelector('#replay-status')?.textContent ?? '',
    playLabel: document.querySelector('#play-toggle')?.textContent ?? '',
    tick: Number(document.querySelector('#replay-scrubber')?.value ?? -1),
  }));

  await page.screenshot({ path: screenshotPath });
  console.log('INITIAL', JSON.stringify(initial));
  console.log('FINAL', JSON.stringify(finalState));
  console.log('REPLAYED', JSON.stringify(replayedState));
  console.log('SCREENSHOT', screenshotPath);
  if (errors.length) console.log('ERRORS', JSON.stringify(errors, null, 2));

  if (errors.length) process.exitCode = 1;
  if (!initial.hasCanvas || initial.canvasWidth <= 0 || initial.canvasHeight <= 0)
    process.exitCode = 2;
  if (!initial.hasWebGl || initial.maxTick !== expected.finalTick) process.exitCode = 3;
  if (
    expected.commanderPlanVersion !== null &&
    (!initial.commanderVisible ||
      !finalState.commanderText.includes(`plan v${expected.commanderPlanVersion}`))
  )
    process.exitCode = 7;
  if (!finalState.status.includes(`winner ${expected.winner}`) || finalState.error)
    process.exitCode = 4;
  if (!(replayedState.tick > 0 && replayedState.tick < initial.maxTick)) process.exitCode = 5;
  if (replayedState.playLabel !== 'Pause') process.exitCode = 6;
} finally {
  await browser?.close();
  await stopServer(ownedServer);
}

async function ensureReplayServer(target) {
  if (await reachable(target)) {
    console.log('REPLAY_SERVER', 'reusing existing server');
    return null;
  }

  const parsed = new URL(target);
  if (!['127.0.0.1', 'localhost'].includes(parsed.hostname)) {
    throw new Error(`Replay server is unreachable: ${target}`);
  }

  const repoRoot = fileURLToPath(new URL('../../', import.meta.url));
  const viteEntry = fileURLToPath(new URL('../../node_modules/vite/bin/vite.js', import.meta.url));
  const port = parsed.port || (parsed.protocol === 'https:' ? '443' : '80');
  let output = '';
  const server = spawn(
    process.execPath,
    [viteEntry, '--host', parsed.hostname, '--port', port, '--strictPort'],
    { cwd: repoRoot, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  server.stdout.on('data', (chunk) => {
    output += String(chunk);
  });
  server.stderr.on('data', (chunk) => {
    output += String(chunk);
  });

  for (let attempt = 0; attempt < 80; attempt++) {
    if (await reachable(target)) {
      console.log('REPLAY_SERVER', `started ${parsed.origin}`);
      return server;
    }
    if (server.exitCode !== null) {
      throw new Error(`Vite exited with code ${server.exitCode}:\n${output}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  server.kill('SIGTERM');
  throw new Error(`Timed out starting Vite at ${parsed.origin}:\n${output}`);
}

async function loadExpectedReplay(target) {
  const pageUrl = new URL(target);
  const recording = pageUrl.searchParams.get('recording');
  if (!recording) throw new Error('Replay URL must include a recording query parameter');
  const response = await fetch(new URL(recording, pageUrl.origin));
  if (!response.ok) throw new Error(`Could not load replay metadata: HTTP ${response.status}`);
  const replay = await response.json();
  const finalTick = replay?.outcome?.finalTick;
  const winner = replay?.outcome?.winner;
  if (!Number.isSafeInteger(finalTick) || !['blue', 'red'].includes(winner)) {
    throw new Error('Replay metadata is missing a terminal tick or winner');
  }
  const tracePath = pageUrl.searchParams.get('trace');
  let commanderPlanVersion = null;
  if (tracePath) {
    const traceResponse = await fetch(new URL(tracePath, pageUrl.origin));
    if (!traceResponse.ok)
      throw new Error(`Could not load commander trace: HTTP ${traceResponse.status}`);
    const trace = await traceResponse.json();
    commanderPlanVersion = trace?.plans?.at(-1)?.version ?? null;
    if (!Number.isSafeInteger(commanderPlanVersion)) {
      throw new Error('Commander trace is missing a final plan version');
    }
  }
  return { finalTick, winner, commanderPlanVersion };
}

async function reachable(target) {
  try {
    const response = await fetch(target, { signal: AbortSignal.timeout(750) });
    return response.ok;
  } catch {
    return false;
  }
}

async function stopServer(server) {
  if (!server || server.exitCode !== null) return;
  server.kill('SIGTERM');
  await Promise.race([
    new Promise((resolve) => server.once('exit', resolve)),
    new Promise((resolve) => setTimeout(resolve, 2_000)),
  ]);
}
