// Installed Electron acceptance driver. Each run owns its home and user-data directory.
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const { _electron } = require(process.argv[2]);
const options = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const input = readline.createInterface({ input: process.stdin });
let app;

// The Python controller interrupts this process when an assertion fails.
// Let Electron close before exiting so the next test can use its native port.
for (const signal of ['SIGINT', 'SIGTERM', ...(process.platform === 'win32' ? ['SIGBREAK'] : [])]) {
  process.once(signal, () => {
    void (async () => {
      try { if (app) await app.close(); }
      finally { process.exit(1); }
    })();
  });
}

function host(action) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Host did not finish ${action}`)), 45000);
    input.once('line', line => { clearTimeout(timer); resolve(JSON.parse(line)); });
    process.stdout.write(JSON.stringify({ action }) + '\n');
  });
}

async function launch() {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({
    executablePath: options.executable,
    args: [`--user-data-dir=${options.userData}`],
    env,
    timeout: 45000,
  });
  app.on('window', () => {
    void app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().forEach(w => w.hide()));
  });
  return app;
}

async function turn(window, label, marker, newSession = true) {
  await window.getByText(label, { exact: true }).waitFor({ timeout: 30000 });
  const composer = window.locator('[contenteditable="true"]');
  if (newSession) {
    const previous = await composer.elementHandle();
    await window.getByRole('button', { name: 'New session', exact: true }).last().click();
    // Native session creation is asynchronous and replaces the session-scoped editor.
    await previous.waitForElementState('hidden');
    await previous.dispose();
  }
  await composer.click();
  await window.keyboard.insertText(`Read the local fixture and reply ${marker}`);
  await window.getByRole('button', { name: 'Send message', exact: true }).click();
  await window.getByText(marker, { exact: true }).waitFor({ timeout: 30000 });
}

(async () => {
  try {
    app = await launch();
    let window = await app.firstWindow();
    if (options.phase === 'initialize') {
      // The shell window can precede profile initialization by a few events.
      const manifest = path.join(process.env.DSH_HOME, 'profiles/desktop/package.json');
      const deadline = Date.now() + 30000;
      while (!fs.existsSync(manifest)) {
        if (Date.now() >= deadline) throw new Error('Desktop did not initialize its profile');
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      return;
    }
    await turn(window, 'FCC Desktop Fixture', 'FCC_DESKTOP_DONE', false);
    await host('refresh');
    await turn(window, 'FCC Updated Fixture', 'FCC_DESKTOP_UPDATED');
    await window.screenshot({ path: path.join(options.artifacts, 'desktop-connected.png') });
    await app.close();
    app = await launch();
    window = await app.firstWindow();
    await turn(window, 'FCC Updated Fixture', 'FCC_DESKTOP_UPDATED');
    await host('disconnect');
    await window.getByRole('button', { name: 'New session', exact: true }).last().click();
    await window.getByText('FCC Updated Fixture', { exact: true }).waitFor({ state: 'hidden', timeout: 30000 });
    await host('reconnect');
    await turn(window, 'FCC Updated Fixture', 'FCC_DESKTOP_UPDATED', false);
    await host('final_disconnect');
    await window.getByRole('button', { name: 'New session', exact: true }).last().click();
    await window.getByText('FCC Updated Fixture', { exact: true }).waitFor({ state: 'hidden', timeout: 30000 });
    await window.screenshot({ path: path.join(options.artifacts, 'desktop-disconnected.png') });
    process.stdout.write(JSON.stringify({ completed: true }) + '\n');
  } finally {
    if (app) await app.close();
    input.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
