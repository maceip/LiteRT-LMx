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

import {css} from 'lit';

/** Shared CSS styles for the chat components. */
export const sharedStyles = css`
  * {
    box-sizing: border-box;
  }

  label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  select, input[type="number"], input[type="text"], textarea {
    background-color: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    color: var(--text);
    font-size: 0.875rem;
    outline: none;
    transition: border-color 0.15s;
    font-family: inherit;
  }

  select:focus, input:focus, textarea:focus {
    border-color: var(--teal);
  }

  /* Base buttons */
  .btn {
    font-weight: 700;
    padding: 12px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-size: 0.875rem;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-family: inherit;
  }

  .btn-primary {
    background-color: var(--teal);
    color: #0b0f19;
  }

  .btn-primary:hover {
    background-color: #00c99ecc;
  }

  .btn-stop {
    background-color: #ef4444 !important;
    color: #ffffff !important;
  }

  .btn-stop:hover {
    background-color: #dc2626 !important;
  }

  .btn-secondary {
    background-color: #1e293b;
    border: 1px solid var(--border);
    color: var(--text);
  }

  .btn-secondary:hover {
    background-color: #334155;
  }

  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
`;
