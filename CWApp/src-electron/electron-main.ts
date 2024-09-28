import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import path from 'path';
import os from 'os';
import childProcess from 'child_process';
import { IS_LINUX, IS_MAC, IS_WINDOWS } from 'app/src-electron/constants';

const noiseAudioPaths: string[] = [];
let firstRunFlag = true;
async function handleFileInput() {
  const dialogResponce = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Audio', extensions: ['wav', 'mp3'] }],
  });
  if (dialogResponce.canceled) {
    return;
  }
  noiseAudioPaths.splice(0, noiseAudioPaths.length);
  noiseAudioPaths.push(...dialogResponce.filePaths);
  return path.basename(dialogResponce.filePaths[0]);
}

async function installLinuxModel() {
  await childProcess.exec(
    'MAMBA_FORCE_BUILD=TRUE conda env create -f src-python/env/linux_environment.yml' +
    '&& conda activate SEMamba',
    (error, stdout, stderr) => {
      if (error !== null) {
        throw error;
      }
      if (stderr.length > 0) {
        console.error(stderr);
        return;
      }
      return stdout;
    }
  );
  await childProcess.exec('conda activate SEMamba');
}

async function installWindowsModel() {
  await childProcess.exec(
    'MAMBA_FORCE_BUILD=TRUE conda env create -f src-python/env/windows_environment.yml' +
    '&& conda activate SEMamba',
    (error, stdout, stderr) => {
      if (error !== null) {
        throw error;
      }
      if (stderr.length > 0) {
        console.error(stderr);
        return;
      }
      return stdout;
    }
  );
}

async function installMacModel() {
  throw new Error('MacOS is not supporting yet')
  // await childProcess.exec(
  //   'MAMBA_FORCE_BUILD=TRUE conda env create -f src-python/env/macos_environment.yml' +
  //   '&& conda activate SEMamba',
  //   (error, stdout, stderr) => {
  //     if (error !== null) {
  //       throw error;
  //     }
  //     if (stderr.length > 0) {
  //       console.error(stderr);
  //       return;
  //     }
  //     return stdout;
  //   }
  // );
}

async function installModel() {
  if (IS_LINUX) {
    return installLinuxModel();
  }
  if (IS_WINDOWS) {
    return installWindowsModel();
  }
  if (IS_MAC) {
    return installMacModel();
  }
}

async function denoiseAudio() {
  return new Promise((resolve, reject) => {
    const audioPath = noiseAudioPaths[0];

    childProcess.exec(
      `python src-python/main.py -i ${audioPath}`,
      (error, stdout, stderr) => {
        if (error !== null) {
          reject(error);
          return;
        }
        if (stderr.length > 0) {
          reject(stderr);
          return;
        }
        resolve({ basename: path.basename(stdout), path: stdout });
      }
    );
  });
}

// needed in case process is undefined under Linux
const platform = process.platform || os.platform();

let mainWindow: BrowserWindow | undefined;

function createWindow() {
  /**
   * Initial window options
   */
  mainWindow = new BrowserWindow({
    icon: path.resolve(__dirname, 'icons/icon.png'), // tray icon
    width: 1000,
    height: 600,
    useContentSize: true,
    webPreferences: {
      contextIsolation: true,
      // More info: https://v2.quasar.dev/quasar-cli-vite/developing-electron-apps/electron-preload-script
      preload: path.resolve(__dirname, process.env.QUASAR_ELECTRON_PRELOAD),
      sandbox: false,
      devTools: true,
      nodeIntegration: true,
      webSecurity: true,
    },
  });
  console.warn('\n\nback is ready\n\n');
  mainWindow.loadURL(process.env.APP_URL);

  if (process.env.DEBUGGING) {
    // if on DEV or Production with debug enabled
    mainWindow.webContents.openDevTools();
  } else {
    // we're on production; no access to devtools pls
    mainWindow.webContents.on('devtools-opened', () => {
      mainWindow?.webContents.closeDevTools();
    });
  }

  mainWindow.on('closed', () => {
    mainWindow = undefined;
  });
}

ipcMain.handle('requestFile', handleFileInput);
ipcMain.handle('denoiseFile', () => {
  if (firstRunFlag) {
    firstRunFlag = !firstRunFlag;
    installModel();
  }
  return denoiseAudio();
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === undefined) {
    createWindow();
  }
});
