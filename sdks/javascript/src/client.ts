import type { Content, QuestionBuilder, SystemOneResponse } from "./types.ts";

export interface SystemClientOptions {
  baseUrl?: string;
}

export interface SystemOneRequest<QS extends Record<string, QuestionBuilder>> {
  state: Content;
  questions: QS;
}

export class SystemClient {
  private readonly baseUrl: string;

  constructor(options: SystemClientOptions = {}) {
    this.baseUrl = (
      options.baseUrl ?? process.env.SYSTEM_BASE_URL ?? "http://127.0.0.1:8010"
    ).replace(/\/+$/, "");
  }

  async systemOne<QS extends Record<string, QuestionBuilder>>(
    request: SystemOneRequest<QS>,
  ): Promise<SystemOneResponse<QS>> {
    const response = await fetch(`${this.baseUrl}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state: request.state,
        questions: Object.fromEntries(
          Object.entries(request.questions).map(([name, builder]) => [name, builder.question]),
        ),
      }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) {
      throw new Error(
        `systemOne failed: HTTP ${response.status} ${(await response.text()).slice(0, 200)}`,
      );
    }
    return (await response.json()) as SystemOneResponse<QS>;
  }
}
