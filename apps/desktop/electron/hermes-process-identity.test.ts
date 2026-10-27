// hermes-process-identity.test.ts
//
// Unit contract for the Windows command-line probe behind isHermesProcess
// (#145235): the win32 path must read Win32_Process.CommandLine through
// powershell.exe (never `ps`, which fails on Windows), report true for a live
// hermes gateway pid and false for a recycled non-hermes pid.
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { windowsHermesProcess } from './hermes-process-identity'

test('win32 probe reads Win32_Process.CommandLine via powershell and never invokes ps', () => {
  const calls: Array<{ command: string; args: string[] }> = []

  const result = windowsHermesProcess(4242, (command, args) => {
    calls.push({ command, args })

    return 'C:\\Python312\\python.exe -m hermes.cli gateway run --profile main'
  })

  assert.equal(result, true)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].command, 'powershell.exe')
  assert.deepEqual(calls[0].args.slice(0, 3), ['-NoProfile', '-NonInteractive', '-Command'])
  assert.equal(
    calls[0].args.some((arg) => arg.includes('Win32_Process')),
    true
  )
  assert.equal(
    calls[0].args.some((arg) => arg.trim().startsWith('ps')),
    false
  )
})

test('win32 probe distinguishes a hermes gateway pid from a recycled non-hermes pid', () => {
  const gateway = windowsHermesProcess(111, () => 'python.exe C:\\apps\\hermes-agent\\hermes gateway run')
  const recycled = windowsHermesProcess(111, () => 'C:\\Windows\\system32\\notepad.exe C:\\notes.txt')

  const probeFailure = windowsHermesProcess(111, () => {
    throw new Error('Get-CimInstance : No match')
  })

  assert.equal(gateway, true)
  assert.equal(recycled, false)
  assert.equal(probeFailure, false)
})
