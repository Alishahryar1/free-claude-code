import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const API_KEY_ENV = "FCC_PI_API_KEY";
const BASE_URL_ENV = "FCC_PI_BASE_URL";
const CATALOG_TIMEOUT_MS = 35000;
const DEFAULT_CONTEXT_WINDOW = 128000;
const DEFAULT_MAX_TOKENS = 16384;

function requireEnvironment(name: string): string {
	const value = process.env[name]?.trim();
	if (!value) {
		throw new Error(`Missing required ${name} environment variable.`);
	}
	return value;
}

function normalizeBaseUrl(value: string): string {
	let url: URL;
	try {
		url = new URL(value);
	} catch {
		throw new Error(`${BASE_URL_ENV} is not a valid URL.`);
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error(`${BASE_URL_ENV} must use http or https.`);
	}
	url.search = "";
	url.hash = "";
	return url.toString().replace(/\/+$/, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalBoolean(value: unknown): boolean | undefined {
	return typeof value === "boolean" ? value : undefined;
}

function optionalPositiveInteger(value: unknown): number | undefined {
	return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
}

function inputModalities(value: unknown): ("text" | "image")[] | undefined {
	if (!Array.isArray(value) || value.length === 0) return undefined;
	if (value.some((item) => item !== "text" && item !== "image")) return undefined;
	const modalities = (["text", "image"] as const).filter((item) => value.includes(item));
	return modalities.includes("text") ? modalities : undefined;
}

function modelDefinition(
	id: string,
	providerModel: string,
	supportsReasoning: boolean | undefined,
	input: ("text" | "image")[] | undefined,
	contextWindow: number | undefined,
	maxTokens: number | undefined,
): ProviderModelConfig {
	return {
		id,
		name: providerModel,
		reasoning: supportsReasoning ?? true,
		input: input ?? ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: contextWindow ?? DEFAULT_CONTEXT_WINDOW,
		maxTokens: maxTokens ?? DEFAULT_MAX_TOKENS,
	};
}

export function projectFccModels(payload: unknown): ProviderModelConfig[] {
	if (!isRecord(payload) || payload.object !== "list" || !Array.isArray(payload.data)) {
		throw new Error("FCC model catalog returned an invalid response shape.");
	}
	const models: ProviderModelConfig[] = [];
	const seen = new Set<string>();
	for (const entry of payload.data) {
		if (!isRecord(entry) || typeof entry.id !== "string" || typeof entry.provider_model_ref !== "string") continue;
		const id = entry.id;
		const providerModel = entry.provider_model_ref;
		if (!id.trim() || !providerModel.includes("/") || seen.has(id)) continue;
		seen.add(id);
		models.push(
			modelDefinition(
				id,
				providerModel,
				optionalBoolean(entry.supportsReasoning),
				inputModalities(entry.inputModalities),
				optionalPositiveInteger(entry.contextWindow),
				optionalPositiveInteger(entry.maxCompletionTokens),
			),
		);
	}

	if (models.length === 0) {
		throw new Error("FCC model catalog contains no routable provider models.");
	}
	return models;
}

function requestIdSuffix(response: Response): string {
	const requestId = response.headers.get("request-id") ?? response.headers.get("x-request-id");
	return requestId ? ` (request ${requestId})` : "";
}

async function fetchFccModels(baseUrl: string, apiKey: string): Promise<ProviderModelConfig[]> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), CATALOG_TIMEOUT_MS);
	try {
		let response: Response;
		try {
			response = await fetch(`${baseUrl}/v1/models?view=messages`, {
				headers: { Authorization: `Bearer ${apiKey}` },
				signal: controller.signal,
			});
		} catch (error) {
			if (error instanceof Error && error.name === "AbortError") {
				throw new Error(`FCC model catalog timed out after ${CATALOG_TIMEOUT_MS}ms.`);
			}
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`Could not reach the FCC model catalog: ${message}`);
		}

		if (!response.ok) {
			throw new Error(`FCC model catalog returned HTTP ${response.status}${requestIdSuffix(response)}.`);
		}

		let payload: unknown;
		try {
			payload = await response.json();
		} catch (error) {
			if (error instanceof Error && error.name === "AbortError") {
				throw new Error(`FCC model catalog timed out after ${CATALOG_TIMEOUT_MS}ms.`);
			}
			throw new Error(`FCC model catalog returned invalid JSON${requestIdSuffix(response)}.`);
		}
		return projectFccModels(payload);
	} finally {
		clearTimeout(timeout);
	}
}

export default async function freeClaudeCode(pi: ExtensionAPI): Promise<void> {
	const baseUrl = normalizeBaseUrl(requireEnvironment(BASE_URL_ENV));
	const apiKey = requireEnvironment(API_KEY_ENV);
	const models = await fetchFccModels(baseUrl, apiKey);

	pi.registerProvider("free-claude-code", {
		name: "Free Claude Code",
		baseUrl,
		apiKey: `$${API_KEY_ENV}`,
		authHeader: true,
		api: "anthropic-messages",
		models,
	});

	pi.on("before_provider_headers", (event, ctx) => {
		if (ctx.model?.provider === "free-claude-code") {
			event.headers["x-opencode-session"] = ctx.sessionManager.getSessionId();
		}
	});

	pi.on("before_provider_request", (event, ctx) => {
		const payload = event.payload;
		if (
			ctx.model?.provider !== "free-claude-code" ||
			ctx.model.api !== "anthropic-messages" ||
			ctx.model.reasoning === false ||
			!isRecord(payload) || payload.model !== ctx.model.id ||
			!isRecord(payload.thinking) || payload.thinking.type !== "enabled" ||
			optionalPositiveInteger(payload.thinking.budget_tokens) === undefined
		) return;

		const effort = pi.getThinkingLevel();
		if (effort === "off") return;

		// Pi synthesizes a token budget from this level. Send the original effort
		// so FCC can choose the provider's supported thinking mode and encoding.
		const thinking = { ...payload.thinking };
		delete thinking.type;
		delete thinking.budget_tokens;
		return {
			...payload,
			thinking: Object.keys(thinking).length > 0 ? thinking : undefined,
			output_config: {
				...(isRecord(payload.output_config) ? payload.output_config : {}),
				effort,
			},
		};
	});
}
