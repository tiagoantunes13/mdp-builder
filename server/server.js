require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'add_layer',
      description: 'Add a new reward layer to the MDP grid',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Layer name (e.g. Parks, Residential, Commercial)' },
          weight: { type: 'number', description: 'Weight between 0 and 1 (e.g. 0.15 for 15%)' }
        },
        required: ['name', 'weight']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'remove_layer',
      description: 'Remove a layer by name',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Name of the layer to remove' }
        },
        required: ['name']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'set_weight',
      description: 'Change the weight of an existing layer',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Name of the layer' },
          weight: { type: 'number', description: 'New weight between 0 and 1' }
        },
        required: ['name', 'weight']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'toggle_layer',
      description: 'Show or hide a layer',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Name of the layer' },
          visible: { type: 'boolean', description: 'true to show, false to hide' }
        },
        required: ['name', 'visible']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'set_goal',
      description: 'Set the goal cell on the grid. Grid is 20 columns wide (0-19) and 10 rows tall (0-9).',
      parameters: {
        type: 'object',
        properties: {
          row: { type: 'integer', description: 'Row index (0-9)' },
          col: { type: 'integer', description: 'Column index (0-19)' }
        },
        required: ['row', 'col']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'set_discount',
      description: 'Set the discount factor (gamma) for policy iteration',
      parameters: {
        type: 'object',
        properties: {
          value: { type: 'number', description: 'Discount factor between 0.5 and 0.99' }
        },
        required: ['value']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'start_solver',
      description: 'Start running the policy iteration solver'
    }
  },
  {
    type: 'function',
    function: {
      name: 'stop_solver',
      description: 'Pause the policy iteration solver'
    }
  },
  {
    type: 'function',
    function: {
      name: 'reset_solver',
      description: 'Reset the solver (clears policy, value function, and iteration count)'
    }
  }
];

const SYSTEM_PROMPT = `You are an AI assistant for an MDP (Markov Decision Process) Builder application.

The app displays a 20-column x 10-row grid where each cell has a reward value. Multiple reward layers (like Roads, Scenic Views, Traffic, Terrain, Safety) are combined with weighted sums. Users can set a goal cell and run policy iteration to find optimal paths.

You can control the builder using the provided tools. When the user asks you to do something, use the appropriate tool. You can call multiple tools in one response if needed.

When adding layers, common types include: Parks, Residential, Commercial, Industrial, Schools, Hospitals. The weight should be between 0 and 1.

Always respond with a brief, friendly explanation of what you did or what information you can provide. Keep responses concise.

The user will provide the current state of the builder (layers, goal, solver status) with each message so you know what's available.`;

app.post('/api/chat', async (req, res) => {
  try {
    const { messages, appState } = req.body;

    const stateContext = `Current builder state:
- Layers: ${appState.layers.map(l => `${l.name} (weight: ${Math.round(l.weight * 100)}%, ${l.visible ? 'visible' : 'hidden'})`).join(', ')}
- Goal cell: ${appState.goalCell ? `[${appState.goalCell.row}, ${appState.goalCell.col}]` : 'not set'}
- Solver: ${appState.isRunning ? 'running' : 'stopped'}, iteration ${appState.iteration}${appState.converged ? ' (converged)' : ''}
- Discount factor: ${appState.discount}`;

    const openaiMessages = [
      { role: 'system', content: SYSTEM_PROMPT + '\n\n' + stateContext },
      ...messages.map(m => ({ role: m.role, content: m.text }))
    ];

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: openaiMessages,
      tools: TOOLS,
      tool_choice: 'auto'
    });

    const choice = completion.choices[0];
    const actions = [];
    let assistantText = choice.message.content || '';

    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        actions.push({
          tool: toolCall.function.name,
          args: JSON.parse(toolCall.function.arguments)
        });
      }

      if (!assistantText) {
        // If model only returned tool calls with no text, ask for a summary
        const followUp = await openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [
            ...openaiMessages,
            choice.message,
            ...choice.message.tool_calls.map(tc => ({
              role: 'tool',
              tool_call_id: tc.id,
              content: JSON.stringify({ success: true, tool: tc.function.name, args: JSON.parse(tc.function.arguments) })
            }))
          ]
        });
        assistantText = followUp.choices[0].message.content || 'Done!';
      }
    }

    res.json({ text: assistantText, actions });
  } catch (error) {
    console.error('OpenAI error:', error.message);
    res.status(500).json({ error: 'Failed to get AI response. Check your API key.' });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Proxy server running on http://localhost:${PORT}`));
