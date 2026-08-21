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
import {customElement, property, state} from 'lit/decorators.js';
/* tslint:disable:no-new-decorators */

/** A custom dropdown component. */
@customElement('custom-dropdown')
export class CustomDropdown extends LitElement {
  @property({ type: String })
  value = '';

  @state()
  private isOpen = false;



  constructor() {
    super();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    document.removeEventListener('click', this.handleOutsideClick);
  }

  private setOpen(isOpen: boolean) {
    if (this.isOpen === isOpen) {
      return;
    }
    this.isOpen = isOpen;
    if (this.isOpen) {
      setTimeout(() => {
        if (this.isConnected) {
          document.addEventListener('click', this.handleOutsideClick);
        }
      }, 0);
    } else {
      document.removeEventListener('click', this.handleOutsideClick);
    }
  }

  private handleOutsideClick = (e: MouseEvent) => {
    if (!this.contains(e.target as Node)) {
      this.setOpen(false);
    }
  };

  static override styles = css`
    :host {
      display: block;
      position: relative;
      font-family: inherit;
    }
    
    .dropdown-button {
      width: 100%;
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid var(--border, #24324f);
      background-color: var(--bg-input, #070a12);
      color: var(--text, #f1f5f9);
      cursor: pointer;
      text-align: left;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.72rem;
      transition: border-color 0.15s, background-color 0.15s;
    }
    
    .dropdown-button:hover {
      border-color: var(--teal, #00c99e);
    }
    
    .dropdown-button:after {
      content: '▼';
      font-size: 0.65rem;
      color: var(--text-muted, #64748b);
      transition: transform 0.2s;
    }

    .dropdown-button.open:after {
      transform: rotate(-180deg);
    }
    
    .dropdown-content {
      display: none;
      position: absolute;
      top: calc(100% + 4px);
      left: 0;
      background-color: var(--bg-card, #151d30);
      width: 100%;
      box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.45);
      z-index: 100000;
      border-radius: 6px;
      border: 1px solid var(--border, #24324f);
      max-height: 280px;
      overflow-y: auto;
      box-sizing: border-box;
    }
    
    .dropdown-content.show {
      display: block;
    }

    ::slotted(.dropdown-item) {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 0.72rem;
      color: var(--text-muted, #64748b);
      transition: background-color 0.15s, color 0.15s;
    }

    ::slotted(.dropdown-item:hover) {
      background-color: var(--bg-input, #070a12);
      color: var(--text, #f1f5f9);
    }

    ::slotted(.dropdown-item[disabled]) {
      opacity: 0.5;
      cursor: not-allowed;
      background-color: var(--bg-input, #070a12);
    }
  `;

  private toggleDropdown() {
    this.setOpen(!this.isOpen);
  }

  private handleItemClick(e: Event) {
    const target = (e.target as HTMLElement).closest('.dropdown-item') as HTMLElement | null;
    
    // Prevent selections click bubbling if they clicked a nested delete button!
    const clickedDelete = (e.target as HTMLElement).closest('.delete-cache-btn');
    if (clickedDelete) {
      return;
    }

    if (target && target.dataset['value'] && !target.hasAttribute('disabled')) {
      this.value = target.dataset['value'];
      this.setOpen(false);
      this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
    }
  }

  override render() {
    // Find the selected item in the slot to display its text in the button
    const items = Array.from(this.querySelectorAll('.dropdown-item'));
    const selectedItem = items.find(item => (item as HTMLElement).dataset['value'] === this.value);
    
    let buttonText = 'Select...';
    if (selectedItem) {
      const spanEl = selectedItem.querySelector('.model-name');
      buttonText = spanEl ? spanEl.textContent || 'Select...' : selectedItem.textContent || 'Select...';
    }

    return html`
      <button class="dropdown-button ${this.isOpen ? 'open' : ''}" @click=${
        this.toggleDropdown}>
        ${buttonText}
      </button>
      <div class="dropdown-content ${this.isOpen ? 'show' : ''}">
        <slot @slotchange=${this.requestUpdate} @click=${
        this.handleItemClick}></slot>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'custom-dropdown': CustomDropdown;
  }
}
