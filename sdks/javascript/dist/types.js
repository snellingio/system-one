// Question and answer types. Builders carry their option keys so the
// response's answers map is fully inferred from the questions you send.
export function noul(instructions, criteria) {
    const question = criteria
        ? { type: "noul", instructions, criteria }
        : { type: "noul", instructions };
    return { kind: "noul", question };
}
export function choice(instructions, options) {
    return { kind: "choice", question: { type: "choice", instructions, criteria: options } };
}
export function score(instructions, levels) {
    return { kind: "score", question: { type: "score", instructions, criteria: levels } };
}
