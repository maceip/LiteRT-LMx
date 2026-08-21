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

import './chat_window';

import {LlmChatStateController} from '../state_controller.js';
import {ChatSessionStore} from '../stores/chat_session_store.js';
import {ModelLoaderService} from '../stores/model_loader_service.js';
import {SettingsStore} from '../stores/settings_store.js';
import {LitertChatWindow} from './chat_window.js';

describe('litert-chat-window', () => {
  let element: LitertChatWindow;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockSettings: jasmine.SpyObj<SettingsStore>;
  let mockModelLoader: jasmine.SpyObj<ModelLoaderService>;
  let cachedModelsMap: Map<string, number>;

  beforeEach(async () => {
    mockChatSession = jasmine.createSpyObj('ChatSessionStore', [
      'sendMessage',
      'cancelGeneration',
    ]);
    // Initialize properties used during render
    mockChatSession.isGenerating = false;
    mockChatSession.messages = [];

    mockSettings = jasmine.createSpyObj('SettingsStore', ['saveSettings']);
    mockSettings.selectedModelPath = 'path/to/model_file.litertlm';

    cachedModelsMap = new Map<string, number>();
    mockModelLoader = jasmine.createSpyObj('ModelLoaderService', [
      'loadModelWeights',
    ]);
    mockModelLoader.isModelLoading = false;
    mockModelLoader.cachedModels = cachedModelsMap;

    mockState = {
      chatSession: mockChatSession,
      settings: mockSettings,
      modelLoader: mockModelLoader,
      statusText: 'Idle',
      addHost: jasmine.createSpy('addHost'),
      removeHost: jasmine.createSpy('removeHost'),
      requestUpdate: jasmine.createSpy('requestUpdate'),
    } as unknown as jasmine.SpyObj<LlmChatStateController>;

    element = document.createElement('litert-chat-window');
    element.state = mockState;
    document.body.appendChild(element);
    await element.updateComplete;
  });

  afterEach(() => {
    element.remove();
  });

  it('renders quick starters when messages are empty', () => {
    const starters = element.shadowRoot!.querySelectorAll('.btn-starter');
    expect(starters.length).toBe(3);
    expect(starters[0]!.textContent!.trim()).toBe('Explain WebGPU');
  });

  it('does not render quick starters when messages exist', async () => {
    mockChatSession.messages = [
      {role: 'user', text: 'Hi', senderName: 'User'},
    ];
    element.requestUpdate();
    await element.updateComplete;

    const starters = element.shadowRoot!.querySelectorAll('.btn-starter');
    expect(starters.length).toBe(0);

    const bubbles = element.shadowRoot!.querySelectorAll('litert-chat-bubble');
    expect(bubbles.length).toBe(1);
  });

  it('disables input when model is loading', async () => {
    mockModelLoader.isModelLoading = true;
    element.requestUpdate();
    await element.updateComplete;

    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    expect(textarea.disabled).toBeTrue();
    expect(textarea.placeholder).toBe('Model is preparing, please wait...');

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBeTrue();
  });

  it('enables input when model is not loading', () => {
    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    expect(textarea.disabled).toBeFalse();
    expect(textarea.placeholder).toBe('Message LiteRT-LM...');

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBeFalse();
  });

  it('shows "Send" when model is cached', async () => {
    cachedModelsMap.set('model_file.litertlm', 1000);
    element.requestUpdate();
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.textContent!.trim()).toBe('Send');
  });

  it('shows "Download & Send" when model is not cached', () => {
    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.textContent!.trim()).toBe('Download & Send');
  });

  it('shows "Stop" button during generation', async () => {
    mockChatSession.isGenerating = true;
    element.requestUpdate();
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send');
    expect(sendBtn).toBeNull();

    const stopBtn = element.shadowRoot!.querySelector('.btn-stop') as HTMLButtonElement;
    expect(stopBtn).toBeTruthy();
    expect(stopBtn.textContent!.trim()).toBe('Stop');

    stopBtn.click();
    expect(mockChatSession.cancelGeneration).toHaveBeenCalled();
  });

  it('sends message on send button click', async () => {
    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    textarea.value = 'Test prompt';
    
    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    sendBtn.click();
    await element.updateComplete;

    expect(mockChatSession.sendMessage).toHaveBeenCalledWith('Test prompt');
    expect(textarea.value).toBe('');
  });

  it('sends message on Enter key press', async () => {
    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    textarea.value = 'Test prompt Enter';
    
    const event = new KeyboardEvent('keydown', {key: 'Enter', shiftKey: false});
    textarea.dispatchEvent(event);
    await element.updateComplete;

    expect(mockChatSession.sendMessage).toHaveBeenCalledWith('Test prompt Enter');
    expect(textarea.value).toBe('');
  });

  it('does not send message on Shift+Enter key press', async () => {
    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    textarea.value = 'Test prompt Shift Enter';
    
    const event = new KeyboardEvent('keydown', {key: 'Enter', shiftKey: true});
    textarea.dispatchEvent(event);
    await element.updateComplete;

    expect(mockChatSession.sendMessage).not.toHaveBeenCalled();
    expect(textarea.value).toBe('Test prompt Shift Enter');
  });

  it('inserts newline on Ctrl+J key press', async () => {
    const textarea =
        element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    textarea.value = 'Line 1Line 2';
    textarea.selectionStart = 6;
    textarea.selectionEnd = 6;

    const event = new KeyboardEvent('keydown', {key: 'j', ctrlKey: true});
    textarea.dispatchEvent(event);
    await element.updateComplete;

    expect(textarea.value).toBe('Line 1\nLine 2');
    expect(textarea.selectionStart).toBe(7);
    expect(textarea.selectionEnd).toBe(7);
  });

  it('populates input when clicking quick starter', async () => {
    const starters = element.shadowRoot!.querySelectorAll('.btn-starter');
    const firstStarter = starters[0] as HTMLButtonElement;
    
    firstStarter.click();
    await element.updateComplete;

    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    expect(textarea.value).toBe('Explain WebGPU in simple terms.');
    expect(element.shadowRoot!.activeElement).toBe(textarea);
  });

  it('handles edit-prompt event', async () => {
    // Add a message to render a bubble
    mockChatSession.messages = [
      {role: 'assistant', text: 'Some response', senderName: 'Assistant'},
    ];
    element.requestUpdate();
    await element.updateComplete;

    const bubble = element.shadowRoot!.querySelector('litert-chat-bubble');
    expect(bubble).toBeTruthy();

    const textarea = element.shadowRoot!.querySelector('#chat-input') as HTMLTextAreaElement;
    expect(textarea.value).toBe('');

    bubble!.dispatchEvent(new CustomEvent('edit-prompt', {
      detail: {prompt: 'Edited Prompt'},
      bubbles: true,
      composed: true,
    }));
    await element.updateComplete;

    expect(textarea.value).toBe('Edited Prompt');
    expect(element.shadowRoot!.activeElement).toBe(textarea);
  });
});
