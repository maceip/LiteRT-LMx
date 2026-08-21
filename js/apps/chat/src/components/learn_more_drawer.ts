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
import {customElement} from 'lit/decorators.js';

import {sharedStyles} from '../styles/shared_styles.js';
/* tslint:disable:no-new-decorators */

/** Component for the right sidebar drawer with "Learn More" information. */
@customElement('litert-learn-more')
export class LitertLearnMore extends LitElement {
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
      
      .section-title {
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--teal);
        margin: 0 0 12px 0;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
      }
      
      .btn-dismiss-right-drawer {
        display: block;
        background: none;
        border: 1px solid rgba(0, 201, 158, 0.3);
        color: var(--teal);
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: bold;
        padding: 4px 10px;
        cursor: pointer;
        transition: background-color 0.15s, border-color 0.15s, color 0.15s;
        font-family: inherit;
        outline: none;
      }
      
      .btn-dismiss-right-drawer:hover {
        background-color: rgba(0, 201, 158, 0.08);
        border-color: var(--teal);
        color: #ffffff;
      }
      
      a {
        transition: opacity 0.15s;
      }
      
      a:hover {
        opacity: 0.8;
      }
    `
  ];



  private dismissLearnMore() {
    this.dispatchEvent(new CustomEvent('close', {
      bubbles: true,
      composed: true,
    }));
  }

  override render() {

    return html`
      <!-- Top Header Group with Done Button -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 12px; flex-shrink: 0;">
        <h2 class="section-title" style="margin: 0; border: none; padding: 0; text-transform: uppercase; letter-spacing: 0.05em;">Learn More</h2>
        <button class="btn-dismiss-right-drawer" @click=${this.dismissLearnMore}>Done</button>
      </div>

      <!-- Dynamic Scrollable Documentation Body -->
      <div style="flex: 1; overflow-y: auto; padding-right: 4px; padding-bottom: 16px; overscroll-behavior: contain;">
        
        <!-- What is LiteRT-LM Section -->
        <div class="doc-section" style="margin-bottom: 20px;">
          <h3 style="font-size: 0.85rem; color: #ffffff; font-weight: bold; margin: 0 0 8px 0; border-left: 3px solid var(--teal); padding-left: 8px;">What is LiteRT-LM?</h3>
          <p style="font-size: 0.78rem; line-height: 1.5; color: #cbd5e1; margin: 0 0 10px 0; text-align: justify;">
            LiteRT-LM is a production-ready, open-source inference framework designed to deliver high-performance, cross-platform LLM deployments on edge devices.
          </p>
          <ul style="font-size: 0.75rem; line-height: 1.45; color: #cbd5e1; padding-left: 16px; margin: 0 0 12px 0; display: flex; flex-direction: column; gap: 6px;">
            <li><b>Cross-Platform Support:</b> Run on Android, iOS, Web, Desktop, and IoT (e.g. Raspberry Pi).</li>
            <li><b>Multi-Language Support:</b> Build with native APIs for <a href="https://ai.google.dev/edge/litert-lm/android" target="_blank" rel="noopener" style="color: var(--teal); text-decoration: none; font-weight: 600;">Kotlin</a>, <a href="https://ai.google.dev/edge/litert-lm/swift" target="_blank" rel="noopener" style="color: var(--teal); text-decoration: none; font-weight: 600;">Swift</a>, <a href="https://ai.google.dev/edge/litert-lm/python" target="_blank" rel="noopener" style="color: var(--teal); text-decoration: none; font-weight: 600;">Python</a>, <a href="https://ai.google.dev/edge/litert-lm/flutter" target="_blank" rel="noopener" style="color: var(--teal); text-decoration: none; font-weight: 600;">Flutter</a>, and <a href="https://ai.google.dev/edge/litert-lm/js" target="_blank" rel="noopener" style="color: var(--teal); text-decoration: none; font-weight: 600;">JavaScript</a>.</li>
            <li><b>Hardware Acceleration:</b> Get peak performance and system stability by leveraging GPU and NPU accelerators across diverse hardware.</li>
            <li><b>Multi-Modality:</b> Build with LLMs that have vision and audio support.</li>
            <li><b>Tool Use:</b> Function calling support for agentic workflows with constrained decoding for improved accuracy.</li>
            <li><b>Broad Model Support:</b> Run Gemma, Llama, Phi-4, Qwen and more.</li>
          </ul>
          <a href="https://ai.google.dev/edge/litert-lm/overview" target="_blank" rel="noopener" style="display: block; text-align: center; font-size: 0.72rem; color: #ffffff; border: 1px solid var(--teal); padding: 8px 10px; border-radius: 4px; text-decoration: none; font-weight: bold; background-color: rgba(0, 201, 158, 0.04); transition: background-color 0.15s; margin-top: 12px;">
            Learn more about LiteRT-LM
          </a>
        </div>

        <!-- How to Get Started Section -->
        <div class="doc-section" style="margin-bottom: 20px; border-top: 1px solid var(--border); padding-top: 16px;">
          <h3 style="font-size: 0.85rem; color: #ffffff; font-weight: bold; margin: 0 0 8px 0; border-left: 3px solid var(--teal); padding-left: 8px;">How to Get Started</h3>
          <p style="font-size: 0.78rem; line-height: 1.5; color: #cbd5e1; margin: 0 0 12px 0;">
            Ready to integrate LiteRT-LM inside your project? You can install the core npm package and explore the official developer guides:
          </p>
          <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px;">
            <a href="https://www.npmjs.com/package/@litert-lm/core" target="_blank" rel="noopener" style="font-size: 0.72rem; color: var(--teal); text-decoration: none; font-weight: 600; border-bottom: 1px dashed rgba(0, 201, 158, 0.4); padding-bottom: 1px; display: inline-block; width: max-content;">
              Explore the npm package
            </a>
            <a href="https://ai.google.dev/edge/litert-lm/js" target="_blank" rel="noopener" style="font-size: 0.72rem; color: var(--teal); text-decoration: none; font-weight: 600; border-bottom: 1px dashed rgba(0, 201, 158, 0.4); padding-bottom: 1px; display: inline-block; width: max-content;">
              Read the JS developer documentation
            </a>
          </div>
        </div>

        <!-- Source Code Section -->
        <div class="doc-section" style="margin-bottom: 16px; border-top: 1px solid var(--border); padding-top: 16px;">
          <h3 style="font-size: 0.85rem; color: #ffffff; font-weight: bold; margin: 0 0 8px 0; border-left: 3px solid var(--teal); padding-left: 8px;">Source Code</h3>
          <p style="font-size: 0.78rem; line-height: 1.5; color: #cbd5e1; margin: 0 0 12px 0;">
            LiteRT-LM is open-source. Check out the official repository, report bugs, or contribute patches:
          </p>
          <a href="https://github.com/google-ai-edge/LiteRT-LM" target="_blank" rel="noopener" style="display: block; text-align: center; font-size: 0.72rem; color: #ffffff; border: 1px solid var(--border); padding: 8px 10px; border-radius: 4px; text-decoration: none; font-weight: bold; background-color: rgba(255,255,255,0.03); transition: background-color 0.15s;">
            GitHub Repository
          </a>
        </div>

        <!-- Anchored Muted Footer Note -->
        <div class="doc-section" style="border-top: 1px dashed var(--border); padding-top: 12px; margin-top: 24px;">
          <p style="font-size: 0.68rem; line-height: 1.5; color: var(--text-muted); font-style: italic; margin: 0; text-align: center;">
            Want to run standard LiteRT models instead of LLMs? Check out 
            <a href="https://www.npmjs.com/package/@litertjs/core?activeTab=readme" target="_blank" rel="noopener" style="color: var(--teal); text-decoration: none; font-weight: bold; border-bottom: 1px dotted rgba(0, 201, 158, 0.4);">@litertjs/core</a>
          </p>
        </div>

      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-learn-more': LitertLearnMore;
  }
}
