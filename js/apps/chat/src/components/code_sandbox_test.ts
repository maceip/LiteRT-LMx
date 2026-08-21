/**
 * Copyright 2026 The ODML Authors.
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

import {CodeSandbox, ConsoleMessage} from './code_sandbox.js';
import * as util from './util.js';

describe('CodeSandbox', () => {
  let sandbox: CodeSandbox;
  let activeIframe: HTMLIFrameElement | null = null;
  let postMessageSpy: jasmine.Spy | null = null;

  beforeEach(() => {
    activeIframe = null;

    // Spy on setSandboxIframeHtml to capture the iframe and mock its loading
    spyOn(util, 'setSandboxIframeHtml').and.callFake((iframe, html) => {
      activeIframe = iframe;
      // Spy on the contentWindow.postMessage of the captured iframe
      if (iframe.contentWindow) {
        postMessageSpy = spyOn(iframe.contentWindow, 'postMessage').and.stub();
      }

      // Simulate async onload
      setTimeout(() => {
        if (iframe.onload) {
          iframe.onload(new Event('load'));
        }
      }, 0);
    });

    sandbox = new CodeSandbox();
  });

  afterEach(() => {
    sandbox.terminate();
  });

  function triggerIframeMessage(data: unknown) {
    if (!activeIframe) {
      throw new Error('No active iframe to trigger message on');
    }
    const event = new MessageEvent('message', {
      data,
      source: activeIframe.contentWindow,
    });
    window.dispatchEvent(event);
  }

  it('can evaluate simple expressions', async () => {
    const promise = sandbox.run('2 + 3');

    // Wait for setTimeout in the spy to fire onload
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(activeIframe).not.toBeNull();
    expect(postMessageSpy!).toHaveBeenCalledWith('2 + 3', '*');

    triggerIframeMessage({
      resultAsString: '5',
      consoleMessages: [],
    });

    const result = await promise;
    expect(result.resultAsString).toBe('5');
    expect(result.error).toBeUndefined();
    expect(result.consoleMessages).toEqual([]);
  });

  it('captures console.log, console.warn, and console.error', async () => {
    const promise = sandbox.run('console.log("hi");');

    await new Promise(resolve => setTimeout(resolve, 0));

    expect(activeIframe).not.toBeNull();

    const mockLogs: ConsoleMessage[] = [
      {type: 'log', text: 'hi'},
    ];
    triggerIframeMessage({
      resultAsString: 'undefined',
      consoleMessages: mockLogs,
    });

    const result = await promise;
    expect(result.resultAsString).toBe('undefined');
    expect(result.error).toBeUndefined();
    expect(result.consoleMessages).toEqual(mockLogs);
  });

  it('handles runtime throwing standard errors', async () => {
    const promise = sandbox.run('throw new Error()');

    await new Promise(resolve => setTimeout(resolve, 0));

    expect(activeIframe).not.toBeNull();

    const expectedConsoleMessages: ConsoleMessage[] = [
      {type: 'error', text: 'Execution Error: Error: something failed'},
    ];
    triggerIframeMessage({
      error: 'Execution Error: Error: something failed',
      consoleMessages: expectedConsoleMessages,
    });

    const result = await promise;
    expect(result.resultAsString).toBeUndefined();
    expect(result.error).toBe('Execution Error: Error: something failed');
    expect(result.consoleMessages).toEqual(expectedConsoleMessages);
  });

  it('times out loop execution and cleans up the iframe', async () => {
    // 50ms timeout
    const promise = sandbox.run('while(true) {}', 50);

    await new Promise(resolve => setTimeout(resolve, 0));

    expect(activeIframe).not.toBeNull();
    const capturedIframe = activeIframe!;

    // Wait for the timeout to fire
    const result = await promise;
    expect(result.resultAsString).toBeUndefined();
    expect(result.error).toContain('Execution timed out after 50ms');

    // Verify iframe was removed from DOM
    expect(capturedIframe.parentNode).toBeNull();
  });

  it('runs normally after explicit termination', async () => {
    void sandbox.run('15 + 15');
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(activeIframe).not.toBeNull();
    const iframe1 = activeIframe!;

    sandbox.terminate();
    expect(iframe1.parentNode).toBeNull();

    const promise2 = sandbox.run('30 + 30');
    await new Promise(resolve => setTimeout(resolve, 0));

    expect(activeIframe).not.toBeNull();
    const iframe2 = activeIframe!;
    expect(iframe2).not.toBe(iframe1); // Should be a new iframe

    triggerIframeMessage({
      resultAsString: '60',
      consoleMessages: [],
    });

    const result = await promise2;
    expect(result.resultAsString).toBe('60');
    expect(result.error).toBeUndefined();
  });
});

