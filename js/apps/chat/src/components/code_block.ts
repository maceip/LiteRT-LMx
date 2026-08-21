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

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

/* tslint:disable:no-new-decorators */

/**
 * Component representing a code block with copy and interactive preview
 * capabilities.
 */
@customElement('litert-code-block')
export class LitertCodeBlock extends LitElement {
  @property({type: String, attribute: 'base-64-code'}) base64Code = '';

  @property({type: String}) language = '';

  static override styles = css`
    :host {
      display: block;
    }
    
    .code-container {
      margin-top: 12px;
      margin-bottom: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background-color: var(--bg-input);
    }

    .code-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background-color: var(--slate-bubble-user);
      padding: 6px 16px;
      border-bottom: 1px solid var(--border);
      font-family: ui-monospace, monospace;
      font-size: 0.75rem;
      color: var(--text-muted);
      line-height: 1.2;
    }

    .code-lang {
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .btn-preview-code {
      background-color: var(--teal);
      color: #0b0f19;
      border: none;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 0.7rem;
      font-weight: 700;
      cursor: pointer;
      transition: background-color 0.15s;
      line-height: 1;
    }

    .btn-preview-code:hover {
      background-color: #00c99ed9;
    }

    .btn-copy-code {
      background-color: #1e293b;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 0.7rem;
      font-weight: 700;
      cursor: pointer;
      transition: background-color 0.15s, color 0.15s;
      line-height: 1;
    }

    .btn-copy-code:hover {
      background-color: var(--teal);
      color: var(--bg-dark) !important;
    }
  `;

  private handlePreview() {
    this.dispatchEvent(new CustomEvent('preview-html', {
      detail: {base64Code: this.base64Code},
      bubbles: true,
      composed: true,
    }));
  }

  private handleCopy(e: Event) {
    const btn = e.target as HTMLButtonElement;
    if (!this.base64Code) return;
    try {
      const binString = atob(this.base64Code);
      const bytes = Uint8Array.from(binString, (m) => m.codePointAt(0)!);
      const code = new TextDecoder().decode(bytes);

      navigator.clipboard.writeText(code)
          .then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => {
              btn.textContent = 'Copy';
            }, 2000);
          })
          .catch(
              err =>
                  console.error('[LiteRT-LM] Failed to copy code block:', err));
    } catch (e) {
      console.error('[LiteRT-LM] Failed to decode code block content:', e);
    }
  }

  override render() {
    const isHtml = this.language === 'html' || this.language === 'xml';
    return html`
      <div class="code-container">
        <div class="code-header">
          <span class="code-lang">${this.language}</span>
          <div style="display: flex; gap: 6px;">
            ${
        isHtml && this.base64Code ?
            html`
              <button class="btn-preview-code" @click=${
                this.handlePreview}>Preview HTML ⚡</button>
            ` :
            ''}
            <button class="btn-copy-code" @click=${
        this.handleCopy}>Copy</button>
          </div>
        </div>
        <slot></slot>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-code-block': LitertCodeBlock;
  }
}
