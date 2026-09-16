// Question and answer types. Builders carry their option keys so the
// response's answers map is fully inferred from the questions you send.

export type Content = string | Record<string, unknown> | unknown[];

export interface NoulQuestion {
  type: "noul";
  instructions: Content;
  criteria?: { true?: string; false?: string };
}

export interface ChoiceQuestion<K extends string = string> {
  type: "choice";
  instructions: Content;
  criteria: Record<K, Content | null>;
}

export interface ScoreQuestion {
  type: "score";
  instructions: Content;
  criteria: Content[];
}

export interface NoulBuilder {
  readonly kind: "noul";
  readonly question: NoulQuestion;
}

export interface ChoiceBuilder<K extends string> {
  readonly kind: "choice";
  readonly question: ChoiceQuestion<K>;
}

export interface ScoreBuilder {
  readonly kind: "score";
  readonly question: ScoreQuestion;
}

export type QuestionBuilder = NoulBuilder | ChoiceBuilder<string> | ScoreBuilder;

export function noul(
  instructions: Content,
  criteria?: { true?: string; false?: string },
): NoulBuilder {
  const question: NoulQuestion = criteria
    ? { type: "noul", instructions, criteria }
    : { type: "noul", instructions };
  return { kind: "noul", question };
}

export function choice<K extends string>(
  instructions: Content,
  options: { [key in K]: Content | null },
): ChoiceBuilder<K> {
  return { kind: "choice", question: { type: "choice", instructions, criteria: options } };
}

export function score(instructions: Content, levels: Content[]): ScoreBuilder {
  return { kind: "score", question: { type: "score", instructions, criteria: levels } };
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
}

export interface ChoiceAnswer<K extends string = string> {
  type: "choice";
  choice: K;
  probabilities: Record<K, number>;
  confidence: number;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, Content>;
  probabilities: Record<string, number>;
  confidence: number;
}

export type AnswerFor<Q> = Q extends { question: ChoiceQuestion<infer K> }
  ? ChoiceAnswer<K>
  : Q extends { question: NoulQuestion }
    ? NoulAnswer
    : Q extends { question: ScoreQuestion }
      ? ScoreAnswer
      : never;

export type Answers<QS extends Record<string, QuestionBuilder>> = {
  [K in keyof QS]: AnswerFor<QS[K]>;
};

export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

export interface SystemOneResponse<QS extends Record<string, QuestionBuilder>> {
  model: string;
  answers: Answers<QS>;
  usage: Usage;
}
