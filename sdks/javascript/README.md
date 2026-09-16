# JavaScript SDK (local clone)

A local clone of the `system-sdk` API, pointed at the System One Lite server
(`server/src/system_one_lite/api.py`). It builds TypeScript to JavaScript for
Node 22.18 or newer. Calls time out after 30 seconds.

Install it from a System One Lite checkout:

```bash
npm install /path/to/system-one/sdks/javascript
```

## Use

```ts
import { choice, SystemClient } from "system-sdk";

const client = new SystemClient();
const response = await client.systemOne({
  state: { document: "I was charged twice. Please fix this ASAP." },
  questions: {
    category: choice("What is this ticket about?", {
      billing: null,
      technical: null,
      other: null,
    }),
  },
});

console.log(response.answers.category.choice); // answer types are inferred
```

Builders: `choice(question, options)`, `noul(question, criteria?)`,
`score(question, levels)`. Answers carry the full shape from
`docs/api.md`: Choice has `choice`, `probabilities`, `confidence`; Score
adds `score` and `legend`; Noul is a single `noul` number.

## Configuration

- `SYSTEM_BASE_URL` — where to call. Defaults to
  `http://127.0.0.1:8010`.
- You can also pass it to the constructor: `new SystemClient({ baseUrl })`.

## Files

- `src/types.ts` — question builders, answer types, response inference.
- `src/client.ts` — `SystemClient` with `systemOne`.
- `examples/quickstart.ts` — the docs example plus all three question types.

Install development dependencies with `npm install`. Run `npm run typecheck`
and `npm test` before publishing or using a changed package.
