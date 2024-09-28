import os from 'os';

export enum NodePlatform {
  Darwin = 'darwin',
  Win32 = 'win32',
  Linux = 'linux',
  Undefined = 'undefined',
}

const PLATFORM_TYPE =
  process.platform ?? os.platform() ?? NodePlatform.Undefined;

export const IS_MAC = PLATFORM_TYPE === NodePlatform.Darwin;

export const IS_WINDOWS = PLATFORM_TYPE === NodePlatform.Win32;

export const IS_LINUX = PLATFORM_TYPE === NodePlatform.Linux;
