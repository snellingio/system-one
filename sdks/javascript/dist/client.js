export class SystemClient {
    baseUrl;
    constructor(options = {}) {
        this.baseUrl = (options.baseUrl ?? process.env.SYSTEM_BASE_URL ?? "http://127.0.0.1:8010").replace(/\/+$/, "");
    }
    async systemOne(request) {
        const response = await fetch(`${this.baseUrl}/evaluate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                state: request.state,
                questions: Object.fromEntries(Object.entries(request.questions).map(([name, builder]) => [name, builder.question])),
            }),
            signal: AbortSignal.timeout(30_000),
        });
        if (!response.ok) {
            throw new Error(`systemOne failed: HTTP ${response.status} ${(await response.text()).slice(0, 200)}`);
        }
        return (await response.json());
    }
}
