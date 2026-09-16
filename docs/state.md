# State

`state` is everything the model may look at when answering. Questions define
the judgments; state is the evidence. One request carries exactly one state,
and every question in that request is answered against it.

## Formats

Three formats are accepted, and each is rendered to text before the model
sees it:

| Format | Rendered as | Good for |
| --- | --- | --- |
| `string` | passed through unchanged | one message, one document, one passage |
| `object` | pretty-printed JSON (`indent=2`) | named fields, records, anything with structure |
| `array` | pretty-printed JSON | ordered items: chat turns, log lines, list of events |

The renderer is `as_text()` in `server/src/system_one_lite/prompts.py`, and it applies the same
rule to `instructions` and to option and level descriptions: strings pass
through, everything else becomes JSON. JSON keys survive rendering, so names
like `"status"` or `"due_date"` reach the model as visible labels.

## Picking a format

Use a string when there is exactly one piece of text and it speaks for
itself. Use an object as soon as the decision needs more than one piece, so
each part carries a name:

```json
{
  "order": {
    "id": "DB-4471",
    "due": "Friday",
    "status": "in transit",
    "region": "eu-west"
  },
  "history": {
    "late_deliveries_this_month": 1,
    "support_contacts_this_week": 2
  },
  "message": "Still waiting on my order. Where is it?"
}
```

A record, some counts, and a message still count as a single state. Bundle
parts together when the answer needs to compare them; keep parts in separate
requests when they are separate decisions.

## Pointing a question at part of the state

To ask about a single field, name that field in the instructions. Backticks
keep the reference unambiguous:

```json
"questions": {
  "complaint_about_delay": {
    "type": "noul",
    "instructions": "Does `message` complain that the delivery is late, given `order.status` and `order.due`?"
  },
  "repeat_issue": {
    "type": "noul",
    "instructions": "Is `history.late_deliveries_this_month` greater than zero?"
  }
}
```

The engine does not resolve these paths itself — they are words in the prompt
the model reads. Named fields plus explicit references keep multi-part states
answerable.

## What makes a good state

- **Complete.** Every fact the questions depend on. The model cannot look
  anything up; it only sees what you send.
- **Relevant.** Extra material dilutes attention. A state of a few hundred
  words beats a dump of everything you have.
- **Factual.** Things you checked (dates, counts, statuses) belong in the
  state, not inside the question. The question states the judgment; the state
  carries the evidence for it.

## Cost note

Each question gets its own copy of the state in its prompt
([How it works](how-it-works.md)). Ten questions over a 500-token state is
about 5,000 input tokens and zero output tokens. Keep the state as tight as
your questions allow, and split big states only when the questions naturally
fall into groups that need different evidence.
