"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// src-electron/electron-main.ts
var import_electron = require("electron");
var import_path = __toESM(require("path"));
var import_os2 = __toESM(require("os"));
var import_child_process = __toESM(require("child_process"));

// src-electron/constants.ts
var import_os = __toESM(require("os"));
var PLATFORM_TYPE = process.platform ?? import_os.default.platform() ?? "undefined" /* Undefined */;
var IS_MAC = PLATFORM_TYPE === "darwin" /* Darwin */;
var IS_WINDOWS = PLATFORM_TYPE === "win32" /* Win32 */;
var IS_LINUX = PLATFORM_TYPE === "linux" /* Linux */;

// src-electron/electron-main.ts
var noiseAudioPaths = [];
var firstRunFlag = true;
async function handleFileInput() {
  const dialogResponce = await import_electron.dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Audio", extensions: ["wav", "mp3"] }]
  });
  if (dialogResponce.canceled) {
    return;
  }
  noiseAudioPaths.splice(0, noiseAudioPaths.length);
  noiseAudioPaths.push(...dialogResponce.filePaths);
  return import_path.default.basename(dialogResponce.filePaths[0]);
}
async function installLinuxModel() {
  await import_child_process.default.exec(
    "MAMBA_FORCE_BUILD=TRUE conda env create -f src-python/env/linux_environment.yml&& conda activate SEMamba",
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
  await import_child_process.default.exec("conda activate SEMamba");
}
async function installWindowsModel() {
  await import_child_process.default.exec(
    "MAMBA_FORCE_BUILD=TRUE conda env create -f src-python/env/windows_environment.yml&& conda activate SEMamba",
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
  throw new Error("MacOS is not supporting yet");
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
    import_child_process.default.exec(
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
        resolve({ basename: import_path.default.basename(stdout), path: stdout });
      }
    );
  });
}
var platform = process.platform || import_os2.default.platform();
var mainWindow;
function createWindow() {
  mainWindow = new import_electron.BrowserWindow({
    icon: import_path.default.resolve(__dirname, "icons/icon.png"),
    width: 1e3,
    height: 600,
    useContentSize: true,
    webPreferences: {
      contextIsolation: true,
      preload: import_path.default.resolve(__dirname, "/Users/anr/Projects/Study/CourseWork2024/CWApp/.quasar/electron/electron-preload.js"),
      sandbox: false,
      devTools: true,
      nodeIntegration: true,
      webSecurity: true
    }
  });
  console.warn("\n\nback is ready\n\n");
  mainWindow.loadURL("http://localhost:9300");
  if (true) {
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.webContents.on("devtools-opened", () => {
      mainWindow?.webContents.closeDevTools();
    });
  }
  mainWindow.on("closed", () => {
    mainWindow = void 0;
  });
}
import_electron.ipcMain.handle("requestFile", handleFileInput);
import_electron.ipcMain.handle("denoiseFile", () => {
  if (firstRunFlag) {
    firstRunFlag = !firstRunFlag;
    installModel();
  }
  return denoiseAudio();
});
import_electron.app.whenReady().then(createWindow);
import_electron.app.on("window-all-closed", () => {
  if (platform !== "darwin") {
    import_electron.app.quit();
  }
});
import_electron.app.on("activate", () => {
  if (mainWindow === void 0) {
    createWindow();
  }
});
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiLi4vLi4vc3JjLWVsZWN0cm9uL2VsZWN0cm9uLW1haW4udHMiLCAiLi4vLi4vc3JjLWVsZWN0cm9uL2NvbnN0YW50cy50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiaW1wb3J0IHsgYXBwLCBCcm93c2VyV2luZG93LCBkaWFsb2csIGlwY01haW4gfSBmcm9tICdlbGVjdHJvbic7XG5pbXBvcnQgcGF0aCBmcm9tICdwYXRoJztcbmltcG9ydCBvcyBmcm9tICdvcyc7XG5pbXBvcnQgY2hpbGRQcm9jZXNzIGZyb20gJ2NoaWxkX3Byb2Nlc3MnO1xuaW1wb3J0IHsgSVNfTElOVVgsIElTX01BQywgSVNfV0lORE9XUyB9IGZyb20gJ2FwcC9zcmMtZWxlY3Ryb24vY29uc3RhbnRzJztcblxuY29uc3Qgbm9pc2VBdWRpb1BhdGhzOiBzdHJpbmdbXSA9IFtdO1xubGV0IGZpcnN0UnVuRmxhZyA9IHRydWU7XG5hc3luYyBmdW5jdGlvbiBoYW5kbGVGaWxlSW5wdXQoKSB7XG4gIGNvbnN0IGRpYWxvZ1Jlc3BvbmNlID0gYXdhaXQgZGlhbG9nLnNob3dPcGVuRGlhbG9nKHtcbiAgICBwcm9wZXJ0aWVzOiBbJ29wZW5GaWxlJ10sXG4gICAgZmlsdGVyczogW3sgbmFtZTogJ0F1ZGlvJywgZXh0ZW5zaW9uczogWyd3YXYnLCAnbXAzJ10gfV0sXG4gIH0pO1xuICBpZiAoZGlhbG9nUmVzcG9uY2UuY2FuY2VsZWQpIHtcbiAgICByZXR1cm47XG4gIH1cbiAgbm9pc2VBdWRpb1BhdGhzLnNwbGljZSgwLCBub2lzZUF1ZGlvUGF0aHMubGVuZ3RoKTtcbiAgbm9pc2VBdWRpb1BhdGhzLnB1c2goLi4uZGlhbG9nUmVzcG9uY2UuZmlsZVBhdGhzKTtcbiAgcmV0dXJuIHBhdGguYmFzZW5hbWUoZGlhbG9nUmVzcG9uY2UuZmlsZVBhdGhzWzBdKTtcbn1cblxuYXN5bmMgZnVuY3Rpb24gaW5zdGFsbExpbnV4TW9kZWwoKSB7XG4gIGF3YWl0IGNoaWxkUHJvY2Vzcy5leGVjKFxuICAgICdNQU1CQV9GT1JDRV9CVUlMRD1UUlVFIGNvbmRhIGVudiBjcmVhdGUgLWYgc3JjLXB5dGhvbi9lbnYvbGludXhfZW52aXJvbm1lbnQueW1sJyArXG4gICAgJyYmIGNvbmRhIGFjdGl2YXRlIFNFTWFtYmEnLFxuICAgIChlcnJvciwgc3Rkb3V0LCBzdGRlcnIpID0+IHtcbiAgICAgIGlmIChlcnJvciAhPT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgIH1cbiAgICAgIGlmIChzdGRlcnIubGVuZ3RoID4gMCkge1xuICAgICAgICBjb25zb2xlLmVycm9yKHN0ZGVycik7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHJldHVybiBzdGRvdXQ7XG4gICAgfVxuICApO1xuICBhd2FpdCBjaGlsZFByb2Nlc3MuZXhlYygnY29uZGEgYWN0aXZhdGUgU0VNYW1iYScpO1xufVxuXG5hc3luYyBmdW5jdGlvbiBpbnN0YWxsV2luZG93c01vZGVsKCkge1xuICBhd2FpdCBjaGlsZFByb2Nlc3MuZXhlYyhcbiAgICAnTUFNQkFfRk9SQ0VfQlVJTEQ9VFJVRSBjb25kYSBlbnYgY3JlYXRlIC1mIHNyYy1weXRob24vZW52L3dpbmRvd3NfZW52aXJvbm1lbnQueW1sJyArXG4gICAgJyYmIGNvbmRhIGFjdGl2YXRlIFNFTWFtYmEnLFxuICAgIChlcnJvciwgc3Rkb3V0LCBzdGRlcnIpID0+IHtcbiAgICAgIGlmIChlcnJvciAhPT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgIH1cbiAgICAgIGlmIChzdGRlcnIubGVuZ3RoID4gMCkge1xuICAgICAgICBjb25zb2xlLmVycm9yKHN0ZGVycik7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHJldHVybiBzdGRvdXQ7XG4gICAgfVxuICApO1xufVxuXG5hc3luYyBmdW5jdGlvbiBpbnN0YWxsTWFjTW9kZWwoKSB7XG4gIHRocm93IG5ldyBFcnJvcignTWFjT1MgaXMgbm90IHN1cHBvcnRpbmcgeWV0JylcbiAgLy8gYXdhaXQgY2hpbGRQcm9jZXNzLmV4ZWMoXG4gIC8vICAgJ01BTUJBX0ZPUkNFX0JVSUxEPVRSVUUgY29uZGEgZW52IGNyZWF0ZSAtZiBzcmMtcHl0aG9uL2Vudi9tYWNvc19lbnZpcm9ubWVudC55bWwnICtcbiAgLy8gICAnJiYgY29uZGEgYWN0aXZhdGUgU0VNYW1iYScsXG4gIC8vICAgKGVycm9yLCBzdGRvdXQsIHN0ZGVycikgPT4ge1xuICAvLyAgICAgaWYgKGVycm9yICE9PSBudWxsKSB7XG4gIC8vICAgICAgIHRocm93IGVycm9yO1xuICAvLyAgICAgfVxuICAvLyAgICAgaWYgKHN0ZGVyci5sZW5ndGggPiAwKSB7XG4gIC8vICAgICAgIGNvbnNvbGUuZXJyb3Ioc3RkZXJyKTtcbiAgLy8gICAgICAgcmV0dXJuO1xuICAvLyAgICAgfVxuICAvLyAgICAgcmV0dXJuIHN0ZG91dDtcbiAgLy8gICB9XG4gIC8vICk7XG59XG5cbmFzeW5jIGZ1bmN0aW9uIGluc3RhbGxNb2RlbCgpIHtcbiAgaWYgKElTX0xJTlVYKSB7XG4gICAgcmV0dXJuIGluc3RhbGxMaW51eE1vZGVsKCk7XG4gIH1cbiAgaWYgKElTX1dJTkRPV1MpIHtcbiAgICByZXR1cm4gaW5zdGFsbFdpbmRvd3NNb2RlbCgpO1xuICB9XG4gIGlmIChJU19NQUMpIHtcbiAgICByZXR1cm4gaW5zdGFsbE1hY01vZGVsKCk7XG4gIH1cbn1cblxuYXN5bmMgZnVuY3Rpb24gZGVub2lzZUF1ZGlvKCkge1xuICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUsIHJlamVjdCkgPT4ge1xuICAgIGNvbnN0IGF1ZGlvUGF0aCA9IG5vaXNlQXVkaW9QYXRoc1swXTtcblxuICAgIGNoaWxkUHJvY2Vzcy5leGVjKFxuICAgICAgYHB5dGhvbiBzcmMtcHl0aG9uL21haW4ucHkgLWkgJHthdWRpb1BhdGh9YCxcbiAgICAgIChlcnJvciwgc3Rkb3V0LCBzdGRlcnIpID0+IHtcbiAgICAgICAgaWYgKGVycm9yICE9PSBudWxsKSB7XG4gICAgICAgICAgcmVqZWN0KGVycm9yKTtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgaWYgKHN0ZGVyci5sZW5ndGggPiAwKSB7XG4gICAgICAgICAgcmVqZWN0KHN0ZGVycik7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIHJlc29sdmUoeyBiYXNlbmFtZTogcGF0aC5iYXNlbmFtZShzdGRvdXQpLCBwYXRoOiBzdGRvdXQgfSk7XG4gICAgICB9XG4gICAgKTtcbiAgfSk7XG59XG5cbi8vIG5lZWRlZCBpbiBjYXNlIHByb2Nlc3MgaXMgdW5kZWZpbmVkIHVuZGVyIExpbnV4XG5jb25zdCBwbGF0Zm9ybSA9IHByb2Nlc3MucGxhdGZvcm0gfHwgb3MucGxhdGZvcm0oKTtcblxubGV0IG1haW5XaW5kb3c6IEJyb3dzZXJXaW5kb3cgfCB1bmRlZmluZWQ7XG5cbmZ1bmN0aW9uIGNyZWF0ZVdpbmRvdygpIHtcbiAgLyoqXG4gICAqIEluaXRpYWwgd2luZG93IG9wdGlvbnNcbiAgICovXG4gIG1haW5XaW5kb3cgPSBuZXcgQnJvd3NlcldpbmRvdyh7XG4gICAgaWNvbjogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJ2ljb25zL2ljb24ucG5nJyksIC8vIHRyYXkgaWNvblxuICAgIHdpZHRoOiAxMDAwLFxuICAgIGhlaWdodDogNjAwLFxuICAgIHVzZUNvbnRlbnRTaXplOiB0cnVlLFxuICAgIHdlYlByZWZlcmVuY2VzOiB7XG4gICAgICBjb250ZXh0SXNvbGF0aW9uOiB0cnVlLFxuICAgICAgLy8gTW9yZSBpbmZvOiBodHRwczovL3YyLnF1YXNhci5kZXYvcXVhc2FyLWNsaS12aXRlL2RldmVsb3BpbmctZWxlY3Ryb24tYXBwcy9lbGVjdHJvbi1wcmVsb2FkLXNjcmlwdFxuICAgICAgcHJlbG9hZDogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgcHJvY2Vzcy5lbnYuUVVBU0FSX0VMRUNUUk9OX1BSRUxPQUQpLFxuICAgICAgc2FuZGJveDogZmFsc2UsXG4gICAgICBkZXZUb29sczogdHJ1ZSxcbiAgICAgIG5vZGVJbnRlZ3JhdGlvbjogdHJ1ZSxcbiAgICAgIHdlYlNlY3VyaXR5OiB0cnVlLFxuICAgIH0sXG4gIH0pO1xuICBjb25zb2xlLndhcm4oJ1xcblxcbmJhY2sgaXMgcmVhZHlcXG5cXG4nKTtcbiAgbWFpbldpbmRvdy5sb2FkVVJMKHByb2Nlc3MuZW52LkFQUF9VUkwpO1xuXG4gIGlmIChwcm9jZXNzLmVudi5ERUJVR0dJTkcpIHtcbiAgICAvLyBpZiBvbiBERVYgb3IgUHJvZHVjdGlvbiB3aXRoIGRlYnVnIGVuYWJsZWRcbiAgICBtYWluV2luZG93LndlYkNvbnRlbnRzLm9wZW5EZXZUb29scygpO1xuICB9IGVsc2Uge1xuICAgIC8vIHdlJ3JlIG9uIHByb2R1Y3Rpb247IG5vIGFjY2VzcyB0byBkZXZ0b29scyBwbHNcbiAgICBtYWluV2luZG93LndlYkNvbnRlbnRzLm9uKCdkZXZ0b29scy1vcGVuZWQnLCAoKSA9PiB7XG4gICAgICBtYWluV2luZG93Py53ZWJDb250ZW50cy5jbG9zZURldlRvb2xzKCk7XG4gICAgfSk7XG4gIH1cblxuICBtYWluV2luZG93Lm9uKCdjbG9zZWQnLCAoKSA9PiB7XG4gICAgbWFpbldpbmRvdyA9IHVuZGVmaW5lZDtcbiAgfSk7XG59XG5cbmlwY01haW4uaGFuZGxlKCdyZXF1ZXN0RmlsZScsIGhhbmRsZUZpbGVJbnB1dCk7XG5pcGNNYWluLmhhbmRsZSgnZGVub2lzZUZpbGUnLCAoKSA9PiB7XG4gIGlmIChmaXJzdFJ1bkZsYWcpIHtcbiAgICBmaXJzdFJ1bkZsYWcgPSAhZmlyc3RSdW5GbGFnO1xuICAgIGluc3RhbGxNb2RlbCgpO1xuICB9XG4gIHJldHVybiBkZW5vaXNlQXVkaW8oKTtcbn0pO1xuXG5hcHAud2hlblJlYWR5KCkudGhlbihjcmVhdGVXaW5kb3cpO1xuXG5hcHAub24oJ3dpbmRvdy1hbGwtY2xvc2VkJywgKCkgPT4ge1xuICBpZiAocGxhdGZvcm0gIT09ICdkYXJ3aW4nKSB7XG4gICAgYXBwLnF1aXQoKTtcbiAgfVxufSk7XG5cbmFwcC5vbignYWN0aXZhdGUnLCAoKSA9PiB7XG4gIGlmIChtYWluV2luZG93ID09PSB1bmRlZmluZWQpIHtcbiAgICBjcmVhdGVXaW5kb3coKTtcbiAgfVxufSk7XG4iLCAiaW1wb3J0IG9zIGZyb20gJ29zJztcblxuZXhwb3J0IGVudW0gTm9kZVBsYXRmb3JtIHtcbiAgRGFyd2luID0gJ2RhcndpbicsXG4gIFdpbjMyID0gJ3dpbjMyJyxcbiAgTGludXggPSAnbGludXgnLFxuICBVbmRlZmluZWQgPSAndW5kZWZpbmVkJyxcbn1cblxuY29uc3QgUExBVEZPUk1fVFlQRSA9XG4gIHByb2Nlc3MucGxhdGZvcm0gPz8gb3MucGxhdGZvcm0oKSA/PyBOb2RlUGxhdGZvcm0uVW5kZWZpbmVkO1xuXG5leHBvcnQgY29uc3QgSVNfTUFDID0gUExBVEZPUk1fVFlQRSA9PT0gTm9kZVBsYXRmb3JtLkRhcndpbjtcblxuZXhwb3J0IGNvbnN0IElTX1dJTkRPV1MgPSBQTEFURk9STV9UWVBFID09PSBOb2RlUGxhdGZvcm0uV2luMzI7XG5cbmV4cG9ydCBjb25zdCBJU19MSU5VWCA9IFBMQVRGT1JNX1RZUEUgPT09IE5vZGVQbGF0Zm9ybS5MaW51eDtcbiJdLAogICJtYXBwaW5ncyI6ICI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLHNCQUFvRDtBQUNwRCxrQkFBaUI7QUFDakIsSUFBQUEsYUFBZTtBQUNmLDJCQUF5Qjs7O0FDSHpCLGdCQUFlO0FBU2YsSUFBTSxnQkFDSixRQUFRLFlBQVksVUFBQUMsUUFBRyxTQUFTLEtBQUs7QUFFaEMsSUFBTSxTQUFTLGtCQUFrQjtBQUVqQyxJQUFNLGFBQWEsa0JBQWtCO0FBRXJDLElBQU0sV0FBVyxrQkFBa0I7OztBRFYxQyxJQUFNLGtCQUE0QixDQUFDO0FBQ25DLElBQUksZUFBZTtBQUNuQixlQUFlLGtCQUFrQjtBQUMvQixRQUFNLGlCQUFpQixNQUFNLHVCQUFPLGVBQWU7QUFBQSxJQUNqRCxZQUFZLENBQUMsVUFBVTtBQUFBLElBQ3ZCLFNBQVMsQ0FBQyxFQUFFLE1BQU0sU0FBUyxZQUFZLENBQUMsT0FBTyxLQUFLLEVBQUUsQ0FBQztBQUFBLEVBQ3pELENBQUM7QUFDRCxNQUFJLGVBQWUsVUFBVTtBQUMzQjtBQUFBLEVBQ0Y7QUFDQSxrQkFBZ0IsT0FBTyxHQUFHLGdCQUFnQixNQUFNO0FBQ2hELGtCQUFnQixLQUFLLEdBQUcsZUFBZSxTQUFTO0FBQ2hELFNBQU8sWUFBQUMsUUFBSyxTQUFTLGVBQWUsVUFBVSxFQUFFO0FBQ2xEO0FBRUEsZUFBZSxvQkFBb0I7QUFDakMsUUFBTSxxQkFBQUMsUUFBYTtBQUFBLElBQ2pCO0FBQUEsSUFFQSxDQUFDLE9BQU8sUUFBUSxXQUFXO0FBQ3pCLFVBQUksVUFBVSxNQUFNO0FBQ2xCLGNBQU07QUFBQSxNQUNSO0FBQ0EsVUFBSSxPQUFPLFNBQVMsR0FBRztBQUNyQixnQkFBUSxNQUFNLE1BQU07QUFDcEI7QUFBQSxNQUNGO0FBQ0EsYUFBTztBQUFBLElBQ1Q7QUFBQSxFQUNGO0FBQ0EsUUFBTSxxQkFBQUEsUUFBYSxLQUFLLHdCQUF3QjtBQUNsRDtBQUVBLGVBQWUsc0JBQXNCO0FBQ25DLFFBQU0scUJBQUFBLFFBQWE7QUFBQSxJQUNqQjtBQUFBLElBRUEsQ0FBQyxPQUFPLFFBQVEsV0FBVztBQUN6QixVQUFJLFVBQVUsTUFBTTtBQUNsQixjQUFNO0FBQUEsTUFDUjtBQUNBLFVBQUksT0FBTyxTQUFTLEdBQUc7QUFDckIsZ0JBQVEsTUFBTSxNQUFNO0FBQ3BCO0FBQUEsTUFDRjtBQUNBLGFBQU87QUFBQSxJQUNUO0FBQUEsRUFDRjtBQUNGO0FBRUEsZUFBZSxrQkFBa0I7QUFDL0IsUUFBTSxJQUFJLE1BQU0sNkJBQTZCO0FBZS9DO0FBRUEsZUFBZSxlQUFlO0FBQzVCLE1BQUksVUFBVTtBQUNaLFdBQU8sa0JBQWtCO0FBQUEsRUFDM0I7QUFDQSxNQUFJLFlBQVk7QUFDZCxXQUFPLG9CQUFvQjtBQUFBLEVBQzdCO0FBQ0EsTUFBSSxRQUFRO0FBQ1YsV0FBTyxnQkFBZ0I7QUFBQSxFQUN6QjtBQUNGO0FBRUEsZUFBZSxlQUFlO0FBQzVCLFNBQU8sSUFBSSxRQUFRLENBQUMsU0FBUyxXQUFXO0FBQ3RDLFVBQU0sWUFBWSxnQkFBZ0I7QUFFbEMseUJBQUFBLFFBQWE7QUFBQSxNQUNYLGdDQUFnQztBQUFBLE1BQ2hDLENBQUMsT0FBTyxRQUFRLFdBQVc7QUFDekIsWUFBSSxVQUFVLE1BQU07QUFDbEIsaUJBQU8sS0FBSztBQUNaO0FBQUEsUUFDRjtBQUNBLFlBQUksT0FBTyxTQUFTLEdBQUc7QUFDckIsaUJBQU8sTUFBTTtBQUNiO0FBQUEsUUFDRjtBQUNBLGdCQUFRLEVBQUUsVUFBVSxZQUFBRCxRQUFLLFNBQVMsTUFBTSxHQUFHLE1BQU0sT0FBTyxDQUFDO0FBQUEsTUFDM0Q7QUFBQSxJQUNGO0FBQUEsRUFDRixDQUFDO0FBQ0g7QUFHQSxJQUFNLFdBQVcsUUFBUSxZQUFZLFdBQUFFLFFBQUcsU0FBUztBQUVqRCxJQUFJO0FBRUosU0FBUyxlQUFlO0FBSXRCLGVBQWEsSUFBSSw4QkFBYztBQUFBLElBQzdCLE1BQU0sWUFBQUYsUUFBSyxRQUFRLFdBQVcsZ0JBQWdCO0FBQUEsSUFDOUMsT0FBTztBQUFBLElBQ1AsUUFBUTtBQUFBLElBQ1IsZ0JBQWdCO0FBQUEsSUFDaEIsZ0JBQWdCO0FBQUEsTUFDZCxrQkFBa0I7QUFBQSxNQUVsQixTQUFTLFlBQUFBLFFBQUssUUFBUSxXQUFXLHFGQUFtQztBQUFBLE1BQ3BFLFNBQVM7QUFBQSxNQUNULFVBQVU7QUFBQSxNQUNWLGlCQUFpQjtBQUFBLE1BQ2pCLGFBQWE7QUFBQSxJQUNmO0FBQUEsRUFDRixDQUFDO0FBQ0QsVUFBUSxLQUFLLHVCQUF1QjtBQUNwQyxhQUFXLFFBQVEsdUJBQW1CO0FBRXRDLE1BQUksTUFBdUI7QUFFekIsZUFBVyxZQUFZLGFBQWE7QUFBQSxFQUN0QyxPQUFPO0FBRUwsZUFBVyxZQUFZLEdBQUcsbUJBQW1CLE1BQU07QUFDakQsa0JBQVksWUFBWSxjQUFjO0FBQUEsSUFDeEMsQ0FBQztBQUFBLEVBQ0g7QUFFQSxhQUFXLEdBQUcsVUFBVSxNQUFNO0FBQzVCLGlCQUFhO0FBQUEsRUFDZixDQUFDO0FBQ0g7QUFFQSx3QkFBUSxPQUFPLGVBQWUsZUFBZTtBQUM3Qyx3QkFBUSxPQUFPLGVBQWUsTUFBTTtBQUNsQyxNQUFJLGNBQWM7QUFDaEIsbUJBQWUsQ0FBQztBQUNoQixpQkFBYTtBQUFBLEVBQ2Y7QUFDQSxTQUFPLGFBQWE7QUFDdEIsQ0FBQztBQUVELG9CQUFJLFVBQVUsRUFBRSxLQUFLLFlBQVk7QUFFakMsb0JBQUksR0FBRyxxQkFBcUIsTUFBTTtBQUNoQyxNQUFJLGFBQWEsVUFBVTtBQUN6Qix3QkFBSSxLQUFLO0FBQUEsRUFDWDtBQUNGLENBQUM7QUFFRCxvQkFBSSxHQUFHLFlBQVksTUFBTTtBQUN2QixNQUFJLGVBQWUsUUFBVztBQUM1QixpQkFBYTtBQUFBLEVBQ2Y7QUFDRixDQUFDOyIsCiAgIm5hbWVzIjogWyJpbXBvcnRfb3MiLCAib3MiLCAicGF0aCIsICJjaGlsZFByb2Nlc3MiLCAib3MiXQp9Cg==
