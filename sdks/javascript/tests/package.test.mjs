import assert from "node:assert/strict";
import test from "node:test";

import { choice, noul, score, SystemClient } from "system-sdk";

test("the built package exports its public API", () => {
  assert.equal(choice("pick", { yes: null }).question.type, "choice");
  assert.equal(noul("check").question.type, "noul");
  assert.equal(score("rate", ["low", "high"]).question.type, "score");
  assert.equal(typeof SystemClient, "function");
});
