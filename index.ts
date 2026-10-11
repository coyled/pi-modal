/**
 * Modal Shared Inference Provider Extension
 *
 * Auto-discovers models from a Modal shared-inference endpoint's
 * OpenAI-compatible `/v1/models` listing instead of requiring a static
 * model list in models.json.
 *
 * Auth:
 *   MODAL_PROXY_TOKEN  — bearer token (required; also stored by /login modal)
 *   MODAL_BASE_URL     — endpoint override (default: https://inference.us-west.modal.direct/v1)
 *   MODAL_API_KEY      — alternative token variable, consulted after MODAL_PROXY_TOKEN
 *
 * Discovery:
 *   Interactive sessions refresh the catalog from the live endpoint after TUI
 *   startup, and the result is persisted to models-store.json so later runs
 *   (including offline and non-interactive ones) start from the snapshot.
 *   Pi never performs a network refresh in print/--list-models mode, so the
 *   extension also probes `/v1/models` once at load time (short timeout,
 *   silent fallback to the persisted snapshot).
 *
 * Endpoint quirks that /v1/models cannot express (e.g. per-model compat
 * flags) can be supplied through a "modal" entry in models.json via `models`
 * or `modelOverrides`, which Pi composes above this provider.
 *
 * Usage:
 *   pi --extension ~/src/pi-modal
 * or symlink/copy into ~/.pi/agent/extensions/modal/ for automatic loading.
 */

// NOTE: deep subpaths like "@earendil-works/pi-ai/api/openai-completions.lazy"
// only work in source/unbundled pi installs. The bundled Node distribution
// resolves extensions against an embedded module map that only covers the
// pi-ai root, /compat, /oauth, and /providers/all specifiers, so everything
// must come from /compat (which re-exports the lazy API wrappers and index).
import {
	createProvider,
	openAICompletionsApi,
	type ApiKeyAuth,
	type AuthResult,
	type Model,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// =============================================================================
// Constants
// =============================================================================

const PROVIDER_ID = "modal";
const DEFAULT_BASE_URL = "https://inference.us-west.modal.direct/v1";
const STARTUP_PROBE_TIMEOUT_MS = 4000;

// Pi thinking levels, lowest to highest. Modal's `reasoning_options.values`
// entries are matched against these to build the thinkingLevelMap.
const PI_THINKING_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;

// There is deliberately no static fallback model list: Modal deployment IDs
// are generated from user-chosen app/model names, so any hardcoded entry
// would be one account's stale snapshot and would surface phantom models
// when the endpoint is unreachable. Cold starts rely on the load-time probe
// and then on the catalog persisted by interactive refreshes.

// =============================================================================
// /v1/models response shape (Modal extends the OpenAI schema)
// =============================================================================

interface ModalListedModel {
	id: string;
	name?: string;
	base_model_id?: string;
	context_length?: number;
	max_output_length?: number;
	input_modalities?: string[];
	reasoning_options?: { type: string; values?: string[] }[];
}

interface ModalModelsResponse {
	data?: ModalListedModel[];
}

// =============================================================================
// Model mapping
// =============================================================================

function mapThinkingLevels(values: string[]): Model<"openai-completions">["thinkingLevelMap"] {
	const supported = new Set(values);
	const map: Record<string, string | null> = {};
	for (const level of PI_THINKING_LEVELS) {
		map[level] = supported.has(level) ? level : null;
	}
	return map;
}

function toPiModel(listed: ModalListedModel, baseUrl: string): Model<"openai-completions"> {
	const reasoningValues =
		listed.reasoning_options?.find((option) => option.type === "effort")?.values ?? [];
	const reasoning = reasoningValues.length > 0;
	const contextWindow = listed.context_length ?? 128000;
	return {
		id: listed.id,
		name: listed.name ?? listed.base_model_id ?? listed.id,
		api: "openai-completions",
		provider: PROVIDER_ID,
		baseUrl,
		reasoning,
		...(reasoning && { thinkingLevelMap: mapThinkingLevels(reasoningValues) }),
		input: listed.input_modalities?.includes("image") ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: listed.max_output_length ?? contextWindow,
		compat: {
			supportsDeveloperRole: false,
			supportsReasoningEffort: reasoning,
			thinkingFormat: "openai",
		},
	};
}

// =============================================================================
// Discovery
// =============================================================================

function normalizeBaseUrl(url: string): string {
	return url.replace(/\/+$/, "");
}

function envBaseUrl(): string | undefined {
	const value = process.env.MODAL_BASE_URL?.trim();
	return value ? normalizeBaseUrl(value) : undefined;
}

function envApiKey(): string | undefined {
	return process.env.MODAL_PROXY_TOKEN?.trim() || process.env.MODAL_API_KEY?.trim() || undefined;
}

async function listRemoteModels(
	baseUrl: string,
	apiKey: string,
	signal: AbortSignal,
): Promise<Model<"openai-completions">[]> {
	const response = await fetch(`${baseUrl}/models`, {
		headers: { Authorization: `Bearer ${apiKey}` },
		signal,
	});
	if (!response.ok) {
		throw new Error(`Modal /models returned HTTP ${response.status}: ${await response.text()}`);
	}
	const payload = (await response.json()) as ModalModelsResponse;
	if (!Array.isArray(payload.data)) {
		throw new Error("Modal /models returned an unexpected payload");
	}
	return payload.data.map((listed) => toPiModel(listed, baseUrl));
}

async function fetchModalModels(context: RefreshModelsContext): Promise<Model<"openai-completions">[]> {
	const credential = context.credential?.type === "api_key" ? context.credential : undefined;
	const apiKey = credential?.key ?? envApiKey();
	if (!apiKey) return [];
	const baseUrl = credential?.env?.MODAL_BASE_URL ?? envBaseUrl() ?? DEFAULT_BASE_URL;
	return listRemoteModels(baseUrl, apiKey, context.signal);
}

// =============================================================================
// Auth
// =============================================================================

function modalApiKeyAuth(): ApiKeyAuth {
	const resolve = async (input: {
		ctx: { env(name: string): Promise<string | undefined> };
		credential?: { type: "api_key"; key?: string; env?: Record<string, string> };
	}): Promise<AuthResult | undefined> => {
		const key =
			input.credential?.key ??
			(await input.ctx.env("MODAL_PROXY_TOKEN")) ??
			(await input.ctx.env("MODAL_API_KEY"));
		if (!key) return undefined;
		const baseUrl = input.credential?.env?.MODAL_BASE_URL ?? (await input.ctx.env("MODAL_BASE_URL"));
		return {
			auth: { apiKey: key, ...(baseUrl ? { baseUrl: normalizeBaseUrl(baseUrl) } : {}) },
			source: input.credential?.key
				? "stored credential"
				: process.env.MODAL_PROXY_TOKEN
					? "MODAL_PROXY_TOKEN"
					: "MODAL_API_KEY",
		};
	};

	return {
		name: "Modal proxy token",
		login: async (interaction) => {
			const key = (
				await interaction.prompt({ type: "secret", message: "Modal proxy token" })
			).trim();
			if (!key) throw new Error("No token entered");
			const baseUrl = (
				await interaction.prompt({
					type: "text",
					message: "Modal inference base URL",
					placeholder: DEFAULT_BASE_URL,
				})
			).trim();
			return {
				type: "api_key",
				key,
				...(baseUrl ? { env: { MODAL_BASE_URL: normalizeBaseUrl(baseUrl) } } : {}),
			};
		},
		check: async ({ ctx, credential }) => {
			const result = await resolve({ ctx, credential });
			return result ? { source: result.source, type: "api_key" } : undefined;
		},
		resolve,
	};
}

// =============================================================================
// Extension entry point
// =============================================================================

export default async function (pi: ExtensionAPI) {
	// Pi only refreshes provider catalogs over the network in interactive
	// sessions. Probe once here so print/--list-models runs see the live
	// catalog. On failure the baseline stays empty and the persisted snapshot
	// (restored by refreshModels from models-store.json) covers offline runs.
	let baseline: Model<"openai-completions">[] = [];
	const apiKey = envApiKey();
	if (apiKey) {
		try {
			baseline = await listRemoteModels(
				envBaseUrl() ?? DEFAULT_BASE_URL,
				apiKey,
				AbortSignal.timeout(STARTUP_PROBE_TIMEOUT_MS),
			);
		} catch {
			// Offline, slow endpoint, or bad token: start empty and let the
			// persisted snapshot or a later interactive refresh fill in.
		}
	}

	pi.registerProvider(
		createProvider({
			id: PROVIDER_ID,
			name: "Modal",
			baseUrl: DEFAULT_BASE_URL,
			auth: { apiKey: modalApiKeyAuth() },
			models: baseline,
			fetchModels: fetchModalModels,
			api: openAICompletionsApi(),
		}),
	);
}
