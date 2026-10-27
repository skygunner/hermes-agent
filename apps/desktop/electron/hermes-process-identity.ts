/**
 * hermes-process-identity.ts
 *
 * Windows command-line probe behind isHermesProcess (#145235): /proc/<pid>/cmdline
 * and `ps -p <pid> -o args=` both fail on Windows (no ps on PATH; MSYS ps rejects
 * the POSIX -o form), so the probe reads Win32_Process.CommandLine through
 * powershell.exe instead — the same CIM pattern backendCommandForPid uses.
 */

import { execFileSync } from 'node:child_process'

import { hiddenWindowsChildOptions } from './windows-child-options'

export type SyncCommandLineExec = (command: string, args: string[]) => string

const defaultSyncExec: SyncCommandLineExec = (command, args) =>
  String(
    execFileSync(command, args, hiddenWindowsChildOptions({ encoding: 'utf8', timeout: 2000 }))
  )

/**
 * True when a live pid's command line mentions "hermes" on Windows. The exec is
 * injectable so tests can prove the win32 path never invokes ps and that a
 * recycled non-hermes pid is not reported as hermes.
 */
export function windowsHermesProcess(
  pid: number,
  exec: SyncCommandLineExec = defaultSyncExec
): boolean {
  try {
    const out = exec('powershell.exe', [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      `(Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}').CommandLine`,
    ])

    return out.toLowerCase().includes('hermes')
  } catch {
    return false
  }
}
