/**
 * Copyright 2026 The ODML Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {setSandboxIframeHtml} from './util.js';

/**
 * Represents a console message captured during code execution in the sandbox.
 */
// tslint:disable-next-line:interface-over-type-literal
export type ConsoleMessage = {
  /** The type of console message: 'log', 'warn', or 'error'. */
  type: 'log'|'warn'|'error',
  /** The text content of the console message. */
  text: string,
};

/**
 * Represents the result of executing code in the sandbox.
 */
// tslint:disable-next-line:interface-over-type-literal
export type SandboxResult = {
  /** The string representation of the execution result, if successful. */
  resultAsString?: string,
  /** The error message, if execution failed. */
  error?: string,
  /** The list of console messages captured during execution. */
  consoleMessages: ConsoleMessage[],
};

/**
 * A class that executes arbitrary JavaScript code in a sandboxed iframe with a
 * timeout.
 */
export class CodeSandbox {
  private activeIframe: HTMLIFrameElement|null = null;
  private activeCleanup: (() => void)|null = null;
  private static readonly DEFAULT_TIMEOUT_MS = 5000;  // 5 seconds


  /**
   * Executes arbitrary JavaScript code in a sandboxed iframe with a timeout.
   * @param code The JavaScript code string to execute.
   * @param timeoutMs The maximum time (in milliseconds) to allow for execution.
   * Defaults to 5000ms (5 seconds).
   * @returns A Promise that resolves with a SandboxResult object.
   */
  run(code: string, timeoutMs: number = CodeSandbox.DEFAULT_TIMEOUT_MS):
      Promise<SandboxResult> {
    // Terminate any ongoing run to ensure we only have one active sandbox
    this.terminate();

    return new Promise((resolve) => {
      const iframe = document.createElement('iframe');
      iframe.style.display = 'none';

      this.activeIframe = iframe;

      let timeoutId: ReturnType<typeof setTimeout>|null = null;

      const cleanup = () => {
        if (timeoutId !== null) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        window.removeEventListener('message', messageHandler);
        if (iframe.parentNode) {
          iframe.parentNode.removeChild(iframe);
        }
        if (this.activeIframe === iframe) {
          this.activeIframe = null;
          this.activeCleanup = null;
        }
      };

      this.activeCleanup = cleanup;

      const messageHandler = (event: MessageEvent) => {
        if (event.source !== iframe.contentWindow) {
          return;
        }
        cleanup();
        resolve(event.data as SandboxResult);
      };

      window.addEventListener('message', messageHandler);

      // Register onload before setting srcdoc to avoid race conditions
      iframe.onload = () => {
        try {
          iframe.contentWindow!.postMessage(code, '*');
        } catch (e) {
          const err = e instanceof Error ? e : new Error(String(e));
          cleanup();
          resolve({
            error: `Failed to send code to sandbox: ${err.message}`,
            consoleMessages: []
          });
        }
      };

      document.body.appendChild(iframe);

      const runnerHtml = `
        <!DOCTYPE html>
        <html>
        <head>
          <script>
            window.addEventListener('message', (event) => {
              if (event.source !== window.parent) return;

              const code = event.data;
              const consoleMessages = [];

              const originalConsole = {
                log: window.console.log,
                warn: window.console.warn,
                error: window.console.error,
              };

              const captureConsoleMessage = (type, ...args) => {
                const text = args.map(arg => {
                  if (arg instanceof Error) return arg.stack || arg.toString();
                  if (typeof arg === 'object' && arg !== null) {
                    try {
                      return JSON.stringify(arg);
                    } catch (e) {
                      return typeof arg.toString === 'function' ? arg.toString() : '[Unstringifiable Object]';
                    }
                  }
                  return String(arg);
                }).join(' ');
                consoleMessages.push({ type, text });
              };

              window.console.log = (...args) => captureConsoleMessage('log', ...args);
              window.console.warn = (...args) => captureConsoleMessage('warn', ...args);
              window.console.error = (...args) => captureConsoleMessage('error', ...args);

              let resultPayload = {
                resultAsString: undefined,
                error: undefined,
                consoleMessages: consoleMessages,
              };

              try {
                const result = window.eval(code);
                if (typeof result === 'undefined') {
                  resultPayload.resultAsString = 'undefined';
                } else {
                  try {
                    resultPayload.resultAsString = String(result);
                  } catch (e_to_string) {
                    const err = e_to_string instanceof Error ? e_to_string : new Error(String(e_to_string));
                    resultPayload.resultAsString = \`[Error converting result to string: \${err.message}]\`;
                    consoleMessages.push({ type: 'error', text: \`Error converting result to string: \${err.message}\` });
                  }
                }
              } catch (e) {
                const err = e instanceof Error ? e : new Error(String(e));
                resultPayload.error = err.stack || err.toString();
                consoleMessages.push({ type: 'error', text: \`Execution Error: \${resultPayload.error}\` });
              } finally {
                window.console.log = originalConsole.log;
                window.console.warn = originalConsole.warn;
                window.console.error = originalConsole.error;
                window.parent.postMessage(resultPayload, '*');
              }
            });
          </script>
        </head>
        <body></body>
        </html>
      `;

      setSandboxIframeHtml(iframe, runnerHtml);

      timeoutId = setTimeout(() => {
        cleanup();
        resolve({
          error: `Execution timed out after ${timeoutMs}ms. The sandbox has been terminated.`,
          consoleMessages: []
        });
      }, timeoutMs);
    });
  }

  /**
   * Terminates the current execution, if active.
   */
  terminate(): void {
    if (this.activeCleanup) {
      this.activeCleanup();
      console.log('Sandbox explicitly terminated.');
    }
  }
}
