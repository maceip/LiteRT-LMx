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

import './chat_bubble';

import {css, html, LitElement} from 'lit';
import {customElement, property, query, state} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** Main chat window component managing messages and input. */
@customElement('litert-chat-window')
export class LitertChatWindow extends LitElement {
  @property({ type: Object })
  state!: LlmChatStateController;

  @state()
  private shouldAutoScroll = true;

  private wasGenerating = false;
  private isProgrammaticScroll = false;

  @query('#chat-messages') private msgBox!: HTMLDivElement;

  @query('#chat-input') private chatInput!: HTMLTextAreaElement;

  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        width: 100%;
        overflow: hidden;
      }

      .chat-messages {
        flex: 1;
        overflow-y: auto;
        overscroll-behavior: contain; 
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        background-color: var(--bg-chat);
      }

      .starters-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 16px 24px 8px 24px;
        background-color: var(--bg-chat);
        flex-shrink: 0;
      }

      .btn-starter {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid var(--border);
        color: var(--text-muted);
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.15s;
      }

      .btn-starter:hover:not(:disabled) {
        background-color: var(--border);
        color: var(--text);
      }

      .chat-input-container {
        padding: 16px 24px 24px 24px;
        background-color: var(--bg-card);
        border-top: 1px solid var(--border);
        display: flex;
        gap: 12px;
        align-items: flex-end;
        flex-shrink: 0;
      }

      .input-textarea-wrapper {
        flex: 1;
        position: relative;
      }

      .btn-input-send {
        height: 40px;
        width: 80px;
        box-sizing: border-box;
        padding: 0;
        flex-shrink: 0;
        font-size: 0.8rem;
        border-radius: 8px;
        margin-bottom: 4px; 
      }
    `
  ];

  override connectedCallback() {
    super.connectedCallback();
    // Register this chat window as a host of our centralized state
    this.state.addHost(this);
  }

  override disconnectedCallback() {
    this.state.removeHost(this);
    super.disconnectedCallback();
  }

  override firstUpdated() {
    // Bind scroll event listener to manage user-driven auto-scroll anchoring
    if (this.msgBox) {
      this.msgBox.addEventListener('scroll', () => {
        // Ignore our own programmatic scrolls that follow the message stream.
        if (this.isProgrammaticScroll) {
          this.isProgrammaticScroll = false;
          return;
        }

        // Otherwise, it was a manual user scroll. Un-anchor if they scroll up.
        const threshold = 6;
        const atBottom = this.msgBox.scrollHeight - this.msgBox.scrollTop -
                this.msgBox.clientHeight <=
            threshold;

        if (!atBottom) {
          this.shouldAutoScroll = false;  // Manual scroll up, pause auto-scroll
        } else {
          this.shouldAutoScroll = true;  // Manual scroll down to bottom, resume
        }
      });
    }
  }

  protected override updated() {
    // Auto-scroll on new message.
    if (this.state.chatSession.isGenerating && !this.wasGenerating) {
      this.shouldAutoScroll = true;
    }
    this.wasGenerating = this.state.chatSession.isGenerating;

    if (this.state.chatSession.isGenerating && this.shouldAutoScroll) {
      requestAnimationFrame(() => {
        if (this.msgBox) {
          this.isProgrammaticScroll = true;
          this.msgBox.scrollTop = this.msgBox.scrollHeight;
        }
      });
    }
  }

  private handleInputHeight(e: Event) {
    const txtarea = e.target as HTMLTextAreaElement;
    txtarea.style.height = '48px'; // Reset to base height first
    const newHeight = Math.max(48, txtarea.scrollHeight);
    txtarea.style.height = Math.min(newHeight, 160) + 'px';
  }

  private handleEditPrompt(e: CustomEvent<{prompt: string}>) {
    const txtarea = this.chatInput;
    if (txtarea) {
      txtarea.value = e.detail.prompt;
      txtarea.focus();
      this.handleInputHeight({target: txtarea} as unknown as Event);
    }
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'j' && e.ctrlKey) {
      // Ctrl+J for a newline (like shift + enter) without sending the message.
      // because CLI apps use Ctrl+J to insert newlines.
      e.preventDefault();
      const target = e.target as HTMLTextAreaElement;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      target.value =
          target.value.substring(0, start) + '\n' + target.value.substring(end);
      target.selectionStart = target.selectionEnd = start + 1;
      this.handleInputHeight(e);
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      // Shift+Enter for a newline, Enter to send the message.
      e.preventDefault();
      this.triggerSendMessage();
    }
  }

  private triggerSendMessage() {
    const txtarea = this.chatInput;
    if (txtarea && txtarea.value.trim() &&
        !this.state.chatSession.isGenerating) {
      const promptText = txtarea.value;
      txtarea.value = '';
      txtarea.style.height = '48px';

      // Trigger centralized message generation
      this.state.chatSession.sendMessage(promptText);
    }
  }

  private useStarter(text: string) {
    const txtarea = this.chatInput;
    if (txtarea) {
      txtarea.value = text;
      txtarea.focus();

      // Trigger manual height adjustments
      this.handleInputHeight({target: txtarea} as unknown as Event);
    }
  }

  override render() {
    const isMessagesEmpty = this.state.chatSession.messages.length === 0;
    const activeModelFilename =
        this.state.settings.selectedModelPath.split('/').pop() || '';
    const isCached =
        this.state.modelLoader.cachedModels.has(activeModelFilename);

    return html`
      <!-- Scrollable Messages Box -->
      <div id="chat-messages" class="chat-messages">

        <!-- Active bubbles mapping -->
        ${this.state.chatSession.messages.map((msg, idx) => html`
          <litert-chat-bubble
            .message=${msg}
            .index=${idx}
            .state=${this.state}
            @edit-prompt=${this.handleEditPrompt}
          ></litert-chat-bubble>
        `)}
      </div>

      <!-- Quick Starters suggestions container (only visible if history is empty) -->
      ${
        isMessagesEmpty ?
            html`
        <div class="starters-container">
          <button class="btn-starter" @click=${
                () => this.useStarter(
                    'Explain WebGPU in simple terms.')}>Explain WebGPU</button>
          <button class="btn-starter" @click=${
                () => this.useStarter(
                    'Write a cool particle effect in Javascript in a single HTML page.')}>Particle Effect</button>
          <button class="btn-starter" @click=${
                () => this.useStarter(
                    'Create a modern dark-themed HTML button with hovering glow transitions.')}>Glow Button HTML</button>
        </div>
      ` :
            ''}


      <!-- Input Area -->
      <div class="chat-input-container">
        <div class="input-wrapper" style="display: flex; flex: 1; width: 100%; gap: 12px; position: relative; align-items: center;">
          <textarea
            id="chat-input"
            placeholder=${
        this.state.modelLoader.isModelLoading ?
            'Model is preparing, please wait...' :
            'Message LiteRT-LM...'}
            ?disabled=${this.state.modelLoader.isModelLoading}
            style="flex: 1; height: 48px; max-height: 160px; background-color: var(--bg-input); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 12px 16px; box-sizing: border-box; font-size: 0.85rem; font-family: inherit; line-height: 1.5; resize: none; transition: border-color 0.15s;"
            @input=${this.handleInputHeight}
            @keydown=${this.handleKeyDown}
          ></textarea>

          <!-- Dynamic Send / Stop action button -->
          ${
        this.state.chatSession.isGenerating ? html`
            <button
              class="btn btn-stop"
              style="height: 38px; padding: 0 16px; font-size: 0.85rem; font-weight: bold; border-radius: 8px; cursor: pointer; transition: background-color 0.15s;"
              @click=${() => this.state.chatSession.cancelGeneration()}
            >
              Stop
            </button>
          ` :
                                              html`
            <button
              id="btn-send"
              class="btn btn-primary"
              ?disabled=${this.state.modelLoader.isModelLoading}
              style="height: 38px; padding: 0 16px; font-size: 0.85rem; font-weight: bold; border-radius: 8px; cursor: pointer; transition: background-color 0.15s;"
              @click=${this.triggerSendMessage}
            >
              ${isCached ? 'Send' : 'Download & Send'}
            </button>
          `}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-chat-window': LitertChatWindow;
  }
}
