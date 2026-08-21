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

import {AutoToolChat, Backend, Engine, LiteRtLm, loadLiteRtLm, ContentPart, ToolWithImplementation, unloadLiteRtLm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

const TEST_TIMEOUT_MS = 1200_000;  // 20 minutes
jasmine.DEFAULT_TIMEOUT_INTERVAL = TEST_TIMEOUT_MS;

const GPU_ARTISAN_MODEL_PATH = '/models/gemma4-e2b-hw.litertlm';

describe('AutoToolChat E2E Integration tests', () => {
  let liteRtLm: LiteRtLm;
  async function resetLiteRtLm() {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
    await liteRtLm.setupDefaultWebGpuDevice();
  }

  beforeAll(async () => {
    await resetLiteRtLm();
  }, TEST_TIMEOUT_MS);

  describe('AutoToolChat E2E', () => {
    let e2eEngine: Engine;

    beforeAll(async () => {
      e2eEngine = await Engine.create({
        model: GPU_ARTISAN_MODEL_PATH,
        backend: Backend.GPU_ARTISAN,
      });
    }, TEST_TIMEOUT_MS);

    afterAll(async () => {
      if (e2eEngine) await e2eEngine.delete();
    });

    it('executes a tool automatically and sees the tool response result',
       async () => {
         const executeSpy =
             jasmine.createSpy('execute').and.resolveTo({temperature: 80});
         const weatherTool: ToolWithImplementation = {
           type: 'function',
           function: {
             name: 'get_weather',
             description: 'Gets weather for a location.',
             parameters: {
               type: 'object',
               properties: {
                 city: {
                   type: 'string',
                   description: 'city name',
                 }
               },
               required: ['city'],
             }
           },
           execute: async (args) => executeSpy(args)
         };

         const conversation = new AutoToolChat({
           engine: e2eEngine,
           tools: [weatherTool],
           config: {
             preface: {
               messages:
                   [{role: 'system', content: 'You are a helpful assistant.'}]
             }
           }
         });

         const message = {
           role: 'user',
           content: 'What is the weather in mountain view?'
         };

         const finalResponse = await conversation.sendMessage(message);

         expect(finalResponse).toBeDefined();
         expect(executeSpy).toHaveBeenCalled();
         expect(finalResponse.content).toBeDefined();

         // Verify that the final text output has accurately injected the
         // tool result payload.
         const textContent = finalResponse.content as ContentPart[];
         const textChunks = textContent.filter(item => item.type === 'text')
                                .map(item => item.text)
                                .join(' ');

         console.error('Final response:', JSON.stringify(finalResponse));
         expect(textChunks).toContain('80');

         await conversation.delete();
       },
       TEST_TIMEOUT_MS);

    it('can execute multiple tool hops sequentially based on logical instructions',
       async () => {
         const getNumberSpy =
             jasmine.createSpy('get_random_number').and.resolveTo({number: 42});
         const numberTool: ToolWithImplementation = {
           type: 'function',
           function: {
             name: 'get_random_number',
             description: 'Generates a random integer.',
             parameters: {
               type: 'object',
               properties: {},
             }
           },
           execute: async (args) => getNumberSpy(args)
         };

         const reportSmallSpy =
             jasmine.createSpy('report_small').and.resolveTo({status: 'ok'});
         const reportSmallTool: ToolWithImplementation = {
           type: 'function',
           function: {
             name: 'report_small',
             description:
                 'Call this function if the random number generated was less than 50.',
             parameters: {
               type: 'object',
               properties: {value: {type: 'integer'}},
               required: ['value']
             }
           },
           execute: async (args) => reportSmallSpy(args)
         };

         const reportLargeSpy =
             jasmine.createSpy('report_large').and.resolveTo({status: 'ok'});
         const reportLargeTool: ToolWithImplementation = {
           type: 'function',
           function: {
             name: 'report_large',
             description:
                 'Call this function if the random number generated was greater than or equal to 50.',
             parameters: {
               type: 'object',
               properties: {value: {type: 'integer'}},
               required: ['value']
             }
           },
           execute: async (args) => reportLargeSpy(args)
         };

         const conversation = new AutoToolChat({
           engine: e2eEngine,
           tools: [numberTool, reportSmallTool, reportLargeTool],
           config: {
             preface: {
               messages: [{
                 role: 'system',
                 content:
                     'You are a test agent. Always follow these rules strictly: 1) Call get_random_number to receive a number. 2) IF the number is < 50, YOU MUST EXCLUSIVELY call the report_small tool providing the number. 3) IF the number is >= 50, YOU MUST EXCLUSIVELY call the report_large tool providing the number. Do not output anything else.'
               }]
             }
           }
         });

         const message = {role: 'user', content: 'Execute your task logic.'};

         // Ensure we allow at least a few recurrences so both tools can execute
         // back to back.
         const finalResponse = await conversation.sendMessage(message);

         expect(finalResponse).toBeDefined();
         expect(getNumberSpy).toHaveBeenCalled();

         // Because we mocked getNumberSpy to return 42, it should ONLY call
         // reportSmallSpy.
         expect(reportSmallSpy).toHaveBeenCalledWith(jasmine.objectContaining({
           value: 42
         }));
         expect(reportLargeSpy).not.toHaveBeenCalled();

         await conversation.delete();
       },
       TEST_TIMEOUT_MS);
  });
});
