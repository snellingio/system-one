// The quickstart example from the SDK docs, against the local server.
// Node 22.18+; run with the server up:
//   node examples/quickstart.ts

import { choice, noul, score, SystemClient } from "../src/index.ts";

const client = new SystemClient();

// The docs' example: answer types are inferred from the questions.
const docs = await client.systemOne({
  state: { document: "I was charged twice. Please fix this ASAP." },
  questions: {
    category: choice("What is this ticket about?", {
      billing: null,
      technical: null,
      other: null,
    }),
  },
});
console.log(docs.answers.category.choice);

// All three types in one call.
const response = await client.systemOne({
  state: { document: "I was charged twice. Please fix this ASAP." },
  questions: {
    billing: noul({ statement: "This ticket is about billing" }),
    tone: choice("What is the customer's tone?", {
      calm: null,
      frustrated: null,
      angry: null,
    }),
    urgency: score({ question: "How urgent is this ticket?" }, [
      { level: "can wait" },
      { level: "this week" },
      { level: "today" },
    ]),
  },
});

console.log(response.answers.billing.noul);
console.log(response.answers.tone.choice);
console.log(response.answers.urgency.score);
console.log(response.usage.output_tokens, "output tokens");
