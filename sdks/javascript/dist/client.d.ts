import type { Content, QuestionBuilder, SystemOneResponse } from "./types.ts";
export interface SystemClientOptions {
    baseUrl?: string;
}
export interface SystemOneRequest<QS extends Record<string, QuestionBuilder>> {
    state: Content;
    questions: QS;
}
export declare class SystemClient {
    private readonly baseUrl;
    constructor(options?: SystemClientOptions);
    systemOne<QS extends Record<string, QuestionBuilder>>(request: SystemOneRequest<QS>): Promise<SystemOneResponse<QS>>;
}
