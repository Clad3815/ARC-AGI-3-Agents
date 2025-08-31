// Reasoning Agent JS (Node) – mimics agents/templates/reasoning_agent.py
// - Uses OpenAI Responses API with gpt-5 and function tools
// - Talks to ARC-AGI-3 API to play a game
// - Sends base64 screenshots generated from the grid (node-canvas)

import 'dotenv/config';
import axios from 'axios';
import { wrapper } from 'axios-cookiejar-support';
import { CookieJar } from 'tough-cookie';
import OpenAI from 'openai';
import { createCanvas } from 'canvas';
import fs from 'fs';

// ---- Config ----
const SCHEME = process.env.SCHEME || 'https';
const HOST = process.env.HOST || 'three.arcprize.org';
const PORT = process.env.PORT || '443';
const ROOT_URL =
  (SCHEME === 'http' && String(PORT) === '80') || (SCHEME === 'https' && String(PORT) === '443')
    ? `${SCHEME}://${HOST}`
    : `${SCHEME}://${HOST}:${PORT}`;

const ARC_API_KEY = process.env.ARC_API_KEY || '';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const DEFAULT_OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5';
const DEFAULT_REASONING_EFFORT = process.env.REASONING_EFFORT || '';
const DEFAULT_GAME_FILTER = process.env.GAME_ID || '';

// Game loop/agent defaults
const MAX_ACTIONS = 400; // align with Python reasoning agent
const MESSAGE_LIMIT = 5; // we recompose each turn; kept for parity
const ZONE_SIZE = 16; // same visual tiling as Python agent
const ACTION_NAME_BY_ID = { 0: 'RESET', 1: 'ACTION1', 2: 'ACTION2', 3: 'ACTION3', 4: 'ACTION4', 5: 'ACTION5', 6: 'ACTION6', 7: 'ACTION7' };

// ---- HTTP clients ----
const jar = new CookieJar();
const arc = wrapper(axios.create({
  baseURL: ROOT_URL,
  headers: {
    'X-API-Key': ARC_API_KEY,
    Accept: 'application/json',
  },
  withCredentials: true,
  jar,
  timeout: 15000,
}));

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// ---- Runtime (for graceful shutdown) ----
const Runtime = {
  cardId: null,
  shuttingDown: false,
};

async function closeScorecardIfOpen(reason = 'shutdown') {
  if (Runtime.shuttingDown) return;
  Runtime.shuttingDown = true;
  try {
    if (Runtime.cardId) {
      const { data } = await arc.post('/api/scorecard/close', { card_id: Runtime.cardId });
      console.log('\n--- SCORECARD REPORT ---');
      console.log(JSON.stringify(data, null, 2));
      console.log(`View your scorecard: ${ROOT_URL}/scorecards/${Runtime.cardId}`);
    } else {
      console.log('No open scorecard to close.');
    }
  } catch (err) {
    console.error('Failed to close scorecard:', err?.response?.data || err.message || err);
    if (Runtime.cardId) {
      console.log(`Scorecard link (may remain open): ${ROOT_URL}/scorecards/${Runtime.cardId}`);
    }
  }
}

function registerShutdownHandlers() {
  const handler = (sig) => {
    console.log(`\nReceived ${sig}, exiting...`);
    closeScorecardIfOpen(sig).finally(() => process.exit(0));
  };
  process.once('SIGINT', handler);
  process.once('SIGTERM', handler);
  process.once('uncaughtException', (err) => {
    console.error('Uncaught exception:', err);
    handler('uncaughtException');
  });
  process.once('unhandledRejection', (reason) => {
    console.error('Unhandled rejection:', reason);
    handler('unhandledRejection');
  });
}

// ---- Helpers: drawing and formatting ----
function gridToPngBase64(grid, cellSize = 40) {
  if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0])) {
    const canvas = createCanvas(200, 200);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 200, 200);
    return canvas.toBuffer('image/png').toString('base64');
  }

  const height = grid.length;
  const width = grid[0].length;
  const canvas = createCanvas(width * cellSize, height * cellSize);
  const ctx = canvas.getContext('2d');

  const keyColors = {
    0: '#FFFFFF',
    1: '#CCCCCC',
    2: '#999999',
    3: '#666666',
    4: '#333333',
    5: '#000000',
    6: '#E53AA3',
    7: '#FF7BCC',
    8: '#F93C31',
    9: '#1E93FF',
    10: '#88D8F1',
    11: '#FFDC00',
    12: '#FF851B',
    13: '#921231',
    14: '#4FCC30',
    15: '#A356D6',
  };

  // draw cells
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = grid[y][x];
      ctx.fillStyle = keyColors[v] || '#888888';
      ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
    }
  }

  // zones and optional labels
  ctx.strokeStyle = '#FFD700';
  ctx.lineWidth = 2;
  ctx.fillStyle = '#FFFFFF';
  ctx.font = `${Math.max(10, Math.floor(cellSize * 0.35))}px sans-serif`;

  for (let y = 0; y < height; y += ZONE_SIZE) {
    for (let x = 0; x < width; x += ZONE_SIZE) {
      const zoneW = Math.min(ZONE_SIZE, width - x) * cellSize;
      const zoneH = Math.min(ZONE_SIZE, height - y) * cellSize;
      ctx.strokeRect(x * cellSize, y * cellSize, zoneW, zoneH);
      // label (try/catch not necessary here, canvas will no-op if font missing)
      ctx.fillText(`(${x},${y})`, x * cellSize + 2, y * cellSize + Math.min(14, cellSize - 2));
    }
  }

  return canvas.toBuffer('image/png').toString('base64');
}

function prettyPrint3D(array3d) {
  if (!Array.isArray(array3d)) return '';
  const lines = [];
  array3d.forEach((block, i) => {
    lines.push(`Grid ${i}:`);
    if (Array.isArray(block)) {
      block.forEach(row => lines.push(`  ${JSON.stringify(row)}`));
    }
    lines.push('');
  });
  return lines.join('\n');
}

// ---- Reasoning tool schema (dynamic allowed actions) ----
function buildReasoningTools(allowedNames = ['ACTION1','ACTION2','ACTION3','ACTION4']) {
  // Base schema
  const properties = {
    reason: {
      type: 'string',
      description: 'Detailed reasoning for choosing this action',
      minLength: 10,
      maxLength: 2000,
    },
    short_description: {
      type: 'string',
      description: 'Brief description of the action',
      minLength: 5,
      maxLength: 500,
    },
    hypothesis: {
      type: 'string',
      description: 'Current hypothesis about game mechanics',
      minLength: 10,
      maxLength: 2000,
    },
    aggregated_findings: {
      type: 'string',
      description: 'Summary of discoveries and learnings so far',
      minLength: 10,
      maxLength: 2000,
    },
    action: {
      type: 'string',
      description: `Choose one of: ${allowedNames.join(', ')}`,
      enum: allowedNames,
    },
  };

  const parameters = {
    type: 'object',
    properties,
    required: ['reason', 'short_description', 'hypothesis', 'aggregated_findings', 'action'],
    additionalProperties: false,
  };

  // Conditionally include x,y only when ACTION6 is available
  if (allowedNames.includes('ACTION6')) {
    properties.x = {
      type: ['integer', 'null'],
      description: 'Required when action==ACTION6. X coordinate in [0,63]; otherwise null.',
      minimum: 0,
      maximum: 63,
    };
    properties.y = {
      type: ['integer', 'null'],
      description: 'Required when action==ACTION6. Y coordinate in [0,63]; otherwise null.',
      minimum: 0,
      maximum: 63,
    };

    parameters.required.push('x', 'y');
  }

  return [
    {
      type: 'function',
      name: 'choose_action',
      description:
        `Choose exactly one game action from [${allowedNames.join(', ')}] and justify it.${
          allowedNames.includes('ACTION6') ? ' If ACTION6, include integer x,y in [0,63].' : ''
        }`,
      parameters,
      strict: true,
    },
  ];
}

function actionDocs(names) {
  const docs = {
    RESET: 'Initialize or restarts the game/level state',
    ACTION1: 'Simple action - varies by game (semantically mapped to up)',
    ACTION2: 'Simple action - varies by game (semantically mapped to down)',
    ACTION3: 'Simple action - varies by game (semantically mapped to left)',
    ACTION4: 'Simple action - varies by game (semantically mapped to right)',
    ACTION5: 'Simple action - varies by game (e.g., interact, select, rotate, attach/detach, execute, etc.)',
    ACTION6: 'Complex action requiring x,y coordinates (0-63 range)',
    ACTION7: 'Simple action - Undo (e.g., interact, select)',
  };
  return names.map(n => `- ${n}: ${docs[n] || ''}`).join('\n');
}

function buildSystemPrompt(allowedNames) {
  const ref = actionDocs(allowedNames);
  const coord = allowedNames.includes('ACTION6')
    ? '\n\nProvide coordinates only when selecting ACTION6; otherwise set x and y to null.'
    : '';
  return `You are an intelligent game researcher tasked with reverse-engineering the rules of a complex puzzle game.

## OBJECTIVE
Your primary mission is to:
1. Discover and understand the complete game mechanics through experimentation
2. Document your findings clearly for your research team
3. Develop a comprehensive theory of how the game operates

## GAME CONTEXT
- This is a sophisticated puzzle game that resembles an IQ test or logic challenge
- The game state is represented as a grid-based environment
- You will receive both visual screenshots and raw numerical grid data
- Game mechanics are unknown and must be discovered through systematic testing

## AVAILABLE ACTIONS
You must choose exactly ONE action per turn from the following options:

${ref}
${coord}

## RESEARCH METHODOLOGY
Your systematic approach should follow this framework:

### 1. HYPOTHESIS FORMATION
- Based on current observations, formulate specific testable theories about game mechanics
- Consider UI elements, player representation, interactive objects, and environmental features
- Pay attention to visual indicators like health/lives, energy meters, inventory, score, etc.

### 2. EXPERIMENTAL TESTING
- Design targeted experiments to validate or refute your hypotheses
- Compare before/after screenshots to identify cause-and-effect relationships
- Test edge cases and boundary conditions
- Document unexpected behaviors or anomalies

### 3. KNOWLEDGE SYNTHESIS
- Consolidate confirmed findings into a coherent understanding
- Identify patterns across different game states
- Build a mental model of the game's rule system
- Note interactions between different game elements (walls, doors, keys, collectibles, etc.)

## ANALYSIS FOCUS AREAS
- **UI Elements**: Lives, energy, score, timers, inventory, action counters
- **Player Character**: Appearance, capabilities, movement constraints
- **Environment**: Obstacles, interactive objects, terrain types
- **Game Logic**: Win/lose conditions, progression mechanics, resource management
- **Visual Feedback**: State changes, animations, highlighting, color coding

Remember: Each action provides valuable data. Observe carefully, think systematically, and build your understanding incrementally.

`;
}

// ---- Agent implementation ----
class ReasoningAgentJS {
  constructor({ rootUrl, gameId, cardId, model, effort }) {
    this.rootUrl = rootUrl;
    this.gameId = gameId;
    this.cardId = cardId;
    this.model = model;
    this.reasoningEffort = effort || '';
    this.guid = '';
    this.frames = [];
    this.history = []; // ReasoningActionResponse objects
    this.screenHistory = []; // PNG bytes
    this.maxScreenHistory = 10;
    this.actionCounter = 0;
    this.tokenCounter = 0; // cumulative input+output tokens
    this.lastReasoningTokens = 0;
    this.totalReasoningTokens = 0;
    // Conversation turns history for Responses API:
    // Each turn = [ userMessage, ...assistantOutputItems, toolOutputItem ]
    this.turns = [];
  }

  clearHistory() {
    this.history = [];
    this.screenHistory = [];
  }

  flattenTurns() {
    // Keep only last MESSAGE_LIMIT turns
    const start = Math.max(0, this.turns.length - MESSAGE_LIMIT);
    const slice = this.turns.slice(start);
    return slice.flat();
  }

  async callOpenAI(latestFrame, previousScreenB64, currentScreenB64) {
    // Dynamic allowed actions for this turn
    const allowedIds = Array.isArray(latestFrame?.available_actions) ? latestFrame.available_actions : null;
    const allowedNames = (allowedIds && allowedIds.length)
      ? allowedIds.map((id) => ACTION_NAME_BY_ID[id]).filter(Boolean)
      : ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'];
    // Push "RESET" to the allowed names
    allowedNames.push('RESET');
    console.log(`Allowed actions this turn: ${allowedNames.join(', ')}`);
    const tools = buildReasoningTools(allowedNames);
    const rawGridText = prettyPrint3D(latestFrame?.frame || []);
    const latestAction = this.history.length > 0 ? this.history[this.history.length - 1] : null;

    const contentParts = [];
    const userText = (
      `Attached are the visual screen and raw grid data.\n\n` +
      `Allowed actions this turn: ${allowedNames.join(', ')}\n` +
      `${allowedNames.includes('ACTION6') ? 'If you choose ACTION6, include integer x,y in [0,63].\n' : ''}` +
      `Raw Grid:\n${rawGridText}\n\nWhat should you do next?`
    );
    contentParts.push({ type: 'input_text', text: userText });
    contentParts.push({ type: 'input_image', image_url: `data:image/png;base64,${currentScreenB64}` });

    // Build user message for this turn
    const userMessage = { role: 'user', content: contentParts };

    const messagesInput = [...this.flattenTurns(), userMessage];

    // Save the messages to a JSON file for debug
    fs.writeFileSync('messages.json', JSON.stringify(messagesInput, null, 2));

    const createPayload = {
      model: this.model,
      instructions: buildSystemPrompt(allowedNames),
      tools,
      tool_choice: 'required',
      parallel_tool_calls: false,
      store: true,
      input: messagesInput,
    };

    if (this.reasoningEffort) {
      createPayload.reasoning = { effort: this.reasoningEffort, summary: "auto" };
    }

    const response = await openai.responses.create(createPayload);

    // token usage (Responses API)
    const u = response?.usage || {};
    const inputTokens = u.input_tokens ?? 0;
    const outputTokens = u.output_tokens ?? 0;
    const totalTokens = u.total_tokens ?? (inputTokens + outputTokens);
    const cachedInput = u.input_tokens_details?.cached_tokens ?? 0;
    const reasoningTokens = u.output_tokens_details?.reasoning_tokens ?? 0;
    this.tokenCounter = (this.tokenCounter || 0) + totalTokens;
    this.lastReasoningTokens = reasoningTokens;
    this.totalReasoningTokens = (this.totalReasoningTokens || 0) + reasoningTokens;
    console.log(
      `[${this.gameId}] tokens in=${inputTokens} (cached=${cachedInput}) out=${outputTokens} reason=${reasoningTokens} total=${totalTokens} | cum=${this.tokenCounter}`
    );

    // pick first function_call in output (global tool choose_action)
    let actionName = 'RESET';
    let args = {};
    let functionCallId = null;
    let actionData = null;
    for (const item of response.output || []) {
      if (item.type === 'function_call') {
        try {
          args = item.arguments ? JSON.parse(item.arguments) || {} : {};
        } catch (_) {
          args = {};
        }
        if (typeof args.action === 'string') actionName = args.action;
        functionCallId = item.call_id || null;
        if (actionName === 'ACTION6') {
          // Validate and clamp x,y coordinates
          let x = typeof args.x === 'number' ? args.x : null;
          let y = typeof args.y === 'number' ? args.y : null;
          if (x != null && y != null) {
            x = Math.min(63, Math.max(0, Math.trunc(x)));
            y = Math.min(63, Math.max(0, Math.trunc(y)));
            actionData = { x, y };
          } else {
            console.warn(`[${this.gameId}] ACTION6 selected without valid x,y – defaulting to (0,0)`);
            actionData = { x: 0, y: 0 };
          }
        }
        break;
      }
    }

    // attach metadata pieces from args (reason, hypothesis, etc.)
    const meta = {
      model: this.model,
      agent_type: 'reasoning_agent',
      reasoning_effort: this.reasoningEffort || undefined,
      reasoning_tokens: reasoningTokens,
      total_reasoning_tokens: this.totalReasoningTokens,
      tokens: {
        input: inputTokens,
        cached_input: cachedInput,
        output: outputTokens,
        total: totalTokens,
        cumulative_total: this.tokenCounter,
      },
      hypothesis: args.hypothesis || '',
      aggregated_findings: args.aggregated_findings || '',
      response_preview: (args.reason || '').slice(0, 200),
      action_chosen: actionName,
      game_context: {
        score: latestFrame?.score ?? 0,
        state: latestFrame?.state ?? 'NOT_PLAYED',
        action_counter: this.actionCounter,
        frame_count: this.frames.length,
      },
    };

    // keep a history entry (align with ReasoningActionResponse)
    const historyEntry = {
      name: actionName,
      reason: args.reason || '',
      short_description: args.short_description || '',
      hypothesis: args.hypothesis || '',
      aggregated_findings: args.aggregated_findings || '',
    };

    // Return items needed to complete the turn
    return { actionName, meta, historyEntry, actionData, responseOutputItems: response.output, userMessage, functionCallId };
  }

  async doAction(actionName, reasoningMeta, actionData = null) {
    const basePayload = { game_id: this.gameId };
    if (this.guid) basePayload.guid = this.guid;
    if (actionName === 'RESET' && this.cardId) basePayload.card_id = this.cardId;
    if (reasoningMeta) basePayload.reasoning = reasoningMeta;
    if (actionName === 'ACTION6' && actionData) {
      const { x, y } = actionData;
      basePayload.x = x;
      basePayload.y = y;
    }

    const url = `/api/cmd/${actionName}`;
    const { data } = await arc.post(url, basePayload);
    if (data?.guid) this.guid = data.guid;
    return data;
  }

  async loop() {
    // Step 0: first action is RESET (like the Python agent)
    const firstFrame = await this.doAction('RESET', {
      model: this.model,
      agent_type: 'reasoning_agent',
      response_preview: 'Initial reset',
      action_chosen: 'RESET',
      game_context: { score: 0, state: 'NOT_PLAYED', action_counter: 0, frame_count: 0 },
    });
    this.frames.push(firstFrame);
    console.log(`[${this.gameId}] RESET -> score ${firstFrame.score}, state ${firstFrame.state}`);

    // Main loop
    while (this.actionCounter < MAX_ACTIONS) {
      const latest = this.frames[this.frames.length - 1];
      if (latest?.state === 'WIN') {
        console.log(`[${this.gameId}] WIN reached after ${this.actionCounter} actions`);
        break;
      }

      // build visuals from current frame
      const currentGrid = Array.isArray(latest?.frame) && latest.frame.length > 0 ? latest.frame[latest.frame.length - 1] : [];
      const currentB64 = gridToPngBase64(currentGrid);
      const previousB64 = this.screenHistory.length > 0 ? this.screenHistory[this.screenHistory.length - 1] : null;

      // decide next action with OpenAI
      const { actionName, meta, historyEntry, actionData, responseOutputItems, userMessage, functionCallId } = await this.callOpenAI(latest, previousB64, currentB64);

      // persist screen for next turn
      this.screenHistory.push(currentB64);
      if (this.screenHistory.length > 10) this.screenHistory.shift();
      this.history.push(historyEntry);

      // submit action (execute the tool)
      const next = await this.doAction(actionName, meta, actionData);
      this.frames.push(next);
      this.actionCounter += 1;
      console.log(`[${this.gameId}] ${actionName} -> score ${next.score}, state ${next.state}`);

      // Build tool output for history (what we'd pass back to the model)
      const toolOutput = {
        type: 'function_call_output',
        call_id: functionCallId || 'call_0',
        output: JSON.stringify({
          status: 'success',
          action: actionName,
          ...(actionData || {}),
          result: {
            score: next?.score ?? 0,
            state: next?.state ?? '',
            full_reset: next?.full_reset ?? false,
          },
        }),
      };

      // Push the turn: user → assistant items → tool output
      this.turns.push([userMessage, ...(responseOutputItems || []), toolOutput]);
      // Enforce MESSAGE_LIMIT turns
      if (this.turns.length > MESSAGE_LIMIT) {
        this.turns = this.turns.slice(-MESSAGE_LIMIT);
      }

      if (next?.full_reset) {
        this.clearHistory();
      }
    }
  }
}

// ---- Orchestration (single-file runner) ----
function parseArgs(argv) {
  const out = { model: '', effort: '', game: '' };
  const parts = [...argv];
  while (parts.length) {
    const token = parts.shift();
    if (!token) break;
    if (token.startsWith('--')) {
      const [key, maybeVal] = token.split('=');
      const k = key.replace(/^--/, '');
      let v = maybeVal;
      if (v === undefined) {
        // take next if present and not another flag
        if (parts[0] && !parts[0].startsWith('--')) v = parts.shift();
      }
      if (k === 'model' && v) out.model = v;
      if (k === 'effort' && v) out.effort = v;
      if (k === 'game' && v) out.game = v;
    }
  }
  return out;
}

async function main() {
  registerShutdownHandlers();
  if (!ARC_API_KEY) {
    console.error('Missing ARC_API_KEY in environment');
    process.exit(1);
  }
  if (!OPENAI_API_KEY) {
    console.error('Missing OPENAI_API_KEY in environment');
    process.exit(1);
  }

  // CLI args
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const SELECTED_MODEL = args.model || DEFAULT_OPENAI_MODEL;
  const SELECTED_EFFORT = args.effort || DEFAULT_REASONING_EFFORT;
  const GAME_FILTER = args.game || DEFAULT_GAME_FILTER;
  console.log(`Config → model=${SELECTED_MODEL} effort=${SELECTED_EFFORT || 'default'} game_filter=${GAME_FILTER || 'all'}`);

  // 1) open scorecard (tags optional)
  const tags = (process.env.TAGS || '').split(',').filter(Boolean);
  const { data: openRes } = await arc.post('/api/scorecard/open', { tags: ['agent', 'reasoning_agent_js', SELECTED_MODEL, ...(SELECTED_EFFORT ? [SELECTED_EFFORT] : []), ...tags] });
  const cardId = String(openRes.card_id);
  Runtime.cardId = cardId;
  console.log(`Scorecard opened: ${cardId}`);

  try {
    // 2) determine games to play
    const filter = GAME_FILTER; // optional prefix or exact id
    const { data: gamesList } = await arc.get('/api/games');
    const all = (gamesList || []).map(g => g.game_id);
    const games = filter ? all.filter(g => g.startsWith(filter)) : all;
    if (!games.length) throw new Error('No games available (check API key or GAME_ID filter).');

    // 3) run agent sequentially across filtered games (simple, single-threaded)
    for (const gameId of games) {
      console.log(`\n=== Playing ${gameId} ===`);
      const agent = new ReasoningAgentJS({ rootUrl: ROOT_URL, gameId, cardId, model: SELECTED_MODEL, effort: SELECTED_EFFORT });
      await agent.loop();
    }

    // 4) close scorecard and print summary
    if (!Runtime.shuttingDown) {
      const { data: closeRes } = await arc.post('/api/scorecard/close', { card_id: cardId });
      Runtime.shuttingDown = true;
      console.log('\n--- FINAL SCORECARD REPORT ---');
      console.log(JSON.stringify(closeRes, null, 2));
      console.log(`View your scorecard: ${ROOT_URL}/scorecards/${cardId}`);
    }
  } catch (err) {
    console.error('Run failed:', err?.response?.data || err.message || err);
    // try to close scorecard even on failure
    try {
      if (!Runtime.shuttingDown) {
        const { data: closeRes } = await arc.post('/api/scorecard/close', { card_id: cardId });
        Runtime.shuttingDown = true;
        console.log('\n--- SCORECARD (partial) ---');
        console.log(JSON.stringify(closeRes, null, 2));
        console.log(`View your scorecard: ${ROOT_URL}/scorecards/${cardId}`);
      }
    } catch (_) { }
    process.exit(1);
  }
}

main();
