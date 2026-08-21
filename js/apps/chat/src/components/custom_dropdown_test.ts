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

import './custom_dropdown';

import {CustomDropdown} from './custom_dropdown.js';

describe('custom-dropdown', () => {
  let element: CustomDropdown;

  beforeEach(async () => {
    element = document.createElement('custom-dropdown');
    element.innerHTML = `
      <div class="dropdown-item" data-value="option1">Option 1</div>
      <div class="dropdown-item" data-value="option2" disabled>Option 2</div>
      <div class="dropdown-item" data-value="option3">
        <span class="model-name">Option 3 (Fancy)</span>
        <button class="delete-cache-btn">X</button>
      </div>
    `;
    document.body.appendChild(element);
    await element.updateComplete;
  });

  afterEach(() => {
    element.remove();
  });

  it('renders default state', () => {
    const button = element.shadowRoot!.querySelector('.dropdown-button');
    expect(button!.textContent!.trim()).toBe('Select...');

    const content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeFalse();
  });

  it('toggles open on click', async () => {
    const button = element.shadowRoot!.querySelector('.dropdown-button') as HTMLButtonElement;
    button.click();
    await element.updateComplete;

    let content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeTrue();
    expect(button.classList.contains('open')).toBeTrue();

    button.click();
    await element.updateComplete;

    content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeFalse();
    expect(button.classList.contains('open')).toBeFalse();
  });

  it('shows selected item text', async () => {
    element.value = 'option1';
    await element.updateComplete;

    const button = element.shadowRoot!.querySelector('.dropdown-button');
    expect(button!.textContent!.trim()).toBe('Option 1');
  });

  it('shows selected item model-name if present', async () => {
    element.value = 'option3';
    await element.updateComplete;

    const button = element.shadowRoot!.querySelector('.dropdown-button');
    expect(button!.textContent!.trim()).toBe('Option 3 (Fancy)');
  });

  it('selects item and fires change event', async () => {
    // Open first
    const button = element.shadowRoot!.querySelector('.dropdown-button') as HTMLButtonElement;
    button.click();
    await element.updateComplete;

    let changeValue: string | null = null;
    element.addEventListener('change', (e: Event) => {
      changeValue = (e as CustomEvent<string>).detail;
    });

    const item1 = element.querySelector('[data-value="option1"]') as HTMLElement;
    item1.click();
    await element.updateComplete;

    expect(element.value).toBe('option1');
    expect(changeValue as string | null).toBe('option1');

    // Should close after selection
    const content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeFalse();
  });

  it('does not select disabled items', async () => {
    const button = element.shadowRoot!.querySelector('.dropdown-button') as HTMLButtonElement;
    button.click();
    await element.updateComplete;

    let changeFired = false;
    element.addEventListener('change', () => {
      changeFired = true;
    });

    const item2 = element.querySelector('[data-value="option2"]') as HTMLElement;
    item2.click();
    await element.updateComplete;

    expect(element.value).toBe('');
    expect(changeFired).toBeFalse();

    // Should stay open
    const content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeTrue();
  });

  it('does not select or close when clicking delete button', async () => {
    const button = element.shadowRoot!.querySelector('.dropdown-button') as HTMLButtonElement;
    button.click();
    await element.updateComplete;

    let changeFired = false;
    element.addEventListener('change', () => {
      changeFired = true;
    });

    const deleteBtn = element.querySelector('.delete-cache-btn') as HTMLElement;
    deleteBtn.click();
    await element.updateComplete;

    expect(element.value).toBe('');
    expect(changeFired).toBeFalse();

    // Should stay open
    const content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeTrue();
  });

  it('closes when clicking outside', async () => {
    const button = element.shadowRoot!.querySelector('.dropdown-button') as HTMLButtonElement;
    button.click();
    await element.updateComplete;

    // Wait for the setTimeout(..., 0) in setOpen to register the document listener
    await new Promise(resolve => setTimeout(resolve, 0));

    let content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeTrue();

    // Click outside
    const outsideEl = document.createElement('div');
    document.body.appendChild(outsideEl);
    outsideEl.click();
    
    await element.updateComplete;

    content = element.shadowRoot!.querySelector('.dropdown-content');
    expect(content!.classList.contains('show')).toBeFalse();

    outsideEl.remove();
  });
});
