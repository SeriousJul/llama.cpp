/** Metadata for server options accepted in model preset INI sections. */

export type PresetArgType = 'string' | 'number' | 'boolean' | 'list-string';

export type PresetCategory =
	| 'model'
	| 'compute'
	| 'context'
	| 'batching'
	| 'memory'
	| 'kv-cache'
	| 'sampling'
	| 'rope'
	| 'multimodal'
	| 'embedding'
	| 'adapters'
	| 'chat'
	| 'server'
	| 'speculative'
	| 'router'
	| 'logging'
	| 'other';

export interface PresetArgCatalogEntry {
	key: string;
	iniKey: string;
	label: string;
	description: string;
	help: string;
	type: PresetArgType;
	category: PresetCategory;
	args: string[];
	valueHint?: string;
	isList?: boolean;
	defaultValue?: string;
	hidden?: boolean;
}

export interface ServerPresetOption {
	key: string;
	args: string[];
	value_hint: string;
	description: string;
	type: 'string' | 'number' | 'boolean';
	sampling: boolean;
	speculative: boolean;
}

export const PRESET_CATEGORY_ORDER: PresetCategory[] = [
	'model',
	'compute',
	'context',
	'batching',
	'memory',
	'kv-cache',
	'sampling',
	'rope',
	'multimodal',
	'embedding',
	'adapters',
	'chat',
	'server',
	'speculative',
	'router',
	'logging',
	'other'
];

export const PRESET_CATEGORIES: Record<PresetCategory, { label: string; description: string }> = {
	adapters: {
		description: 'LoRA adapters, control vectors, and tensor overrides.',
		label: 'Adapters'
	},
	batching: {
		description: 'Batch sizes, slots, and parallel decoding.',
		label: 'Batching and parallelism'
	},
	chat: { description: 'Templates, reasoning, tools, MCP, and agents.', label: 'Chat and tools' },
	compute: { description: 'Thread scheduling, devices, and tensor offload.', label: 'CPU and GPU' },
	context: { description: 'Context size, shifting, and long-context behavior.', label: 'Context' },
	embedding: {
		description: 'Pooling, embedding, and reranking behavior.',
		label: 'Embedding and reranking'
	},
	'kv-cache': {
		description: 'KV cache formats, offload, reuse, and checkpoints.',
		label: 'KV cache'
	},
	logging: { description: 'Logging output and diagnostics.', label: 'Logging' },
	memory: {
		description: 'Model loading, mapping, locking, and placement.',
		label: 'Loading and memory'
	},
	model: {
		description: 'Model files, repositories, aliases, and metadata.',
		label: 'Model source'
	},
	multimodal: { description: 'Projectors and image or media processing.', label: 'Multimodal' },
	other: { description: 'Additional server-supported model options.', label: 'Other' },
	rope: { description: 'Position scaling and context extension.', label: 'RoPE and YaRN' },
	router: { description: 'Per-model loading and shutdown behavior.', label: 'Router lifecycle' },
	sampling: {
		description: 'Token selection, penalties, grammars, and generation limits.',
		label: 'Sampling'
	},
	server: {
		description: 'HTTP behavior, endpoints, timeouts, and Web UI options.',
		label: 'Server runtime'
	},
	speculative: {
		description: 'Draft models and speculative decoding algorithms.',
		label: 'Speculative decoding'
	}
};

const LIST_KEYS = new Set([
	'alias',
	'tags',
	'tools',
	'cors-origins',
	'cors-methods',
	'cors-headers',
	'tensor-split',
	'dry-sequence-breaker'
]);
const GROUPS: Partial<Record<PresetCategory, Set<string>>> = {
	adapters: new Set([
		'lora',
		'lora-scaled',
		'lora-init-without-apply',
		'control-vector',
		'control-vector-scaled',
		'override-kv',
		'override-tensor'
	]),
	batching: new Set(['batch-size', 'ubatch-size', 'parallel', 'cont-batching']),
	context: new Set([
		'ctx-size',
		'n-predict',
		'keep',
		'context-shift',
		'swa-full',
		'swa-checkpoints',
		'checkpoint-min-step'
	]),
	embedding: new Set(['pooling', 'embeddings', 'reranking', 'embd-normalize']),
	'kv-cache': new Set([
		'cache-ram',
		'kv-unified',
		'cache-idle-slots',
		'cache-type-k',
		'cache-type-v',
		'kv-offload',
		'cache-prompt',
		'cache-reuse',
		'defrag-thold'
	]),
	memory: new Set([
		'load-mode',
		'mlock',
		'mmap',
		'direct-io',
		'numa',
		'fit',
		'fit-target',
		'fit-ctx',
		'check-tensors',
		'repack',
		'no-host',
		'op-offload'
	]),
	model: new Set([
		'model',
		'model-url',
		'docker-repo',
		'hf-repo',
		'hf-file',
		'hf-token',
		'alias',
		'tags'
	]),
	multimodal: new Set([
		'mmproj',
		'mmproj-url',
		'mmproj-auto',
		'mmproj-offload',
		'image-min-tokens',
		'image-max-tokens',
		'mtmd-batch-max-tokens',
		'media-path'
	]),
	router: new Set(['load-on-startup', 'stop-timeout']),
	server: new Set(['threads-http'])
};

function categoryFor(option: ServerPresetOption): PresetCategory {
	if (option.speculative || option.key.includes('draft') || option.key.startsWith('spec-'))
		return 'speculative';

	if (option.sampling) return 'sampling';

	if (option.key.startsWith('rope-') || option.key.startsWith('yarn-')) return 'rope';

	if (option.key.startsWith('log-')) return 'logging';

	for (const [category, keys] of Object.entries(GROUPS) as [PresetCategory, Set<string>][]) {
		if (keys.has(option.key)) return category;
	}

	if (
		/^(threads|cpu-|prio|poll|device|n-gpu-layers|split-mode|tensor-split|main-gpu)/.test(
			option.key
		)
	)
		return 'compute';

	if (/^(chat-|reasoning|jinja|tools|mcp-|agent|prefill-assistant|skip-chat)/.test(option.key))
		return 'chat';

	if (
		/^(timeout|sse-|threads-http|cors-|api-prefix|path|webui|metrics|props|slots|slot-|reuse-port|sleep-idle)/.test(
			option.key
		)
	)
		return 'server';

	return 'other';
}

function labelFor(key: string): string {
	return key
		.split('-')
		.map((part) =>
			part.length <= 3 ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1)
		)
		.join(' ');
}

export const PRESET_ARGS_CATALOG: PresetArgCatalogEntry[] = [];

export function setPresetArgsCatalog(options: ServerPresetOption[]): void {
	const entries = options.map((option): PresetArgCatalogEntry => {
		const isList = LIST_KEYS.has(option.key);
		const type = isList ? 'list-string' : option.type;

		return {
			args: option.args,
			category: categoryFor(option),
			defaultValue: type === 'boolean' ? 'false' : '',
			description: option.description.trim(),
			help: option.description.trim(),
			iniKey: option.key,
			isList,
			key: normalizeKey(option.key),
			label: labelFor(option.key),
			type,
			valueHint: option.value_hint || undefined
		};
	});

	PRESET_ARGS_CATALOG.splice(0, PRESET_ARGS_CATALOG.length, ...entries);
}

export function normalizeKey(key: string): string {
	return key.replace(/-/g, '_');
}

export function getPresetArg(key: string): PresetArgCatalogEntry | undefined {
	const normalized = normalizeKey(key);

	return PRESET_ARGS_CATALOG.find((entry) => entry.key === normalized);
}

export function isKnownPresetArg(key: string): boolean {
	return getPresetArg(key) !== undefined;
}

export function getIniKey(key: string): string {
	return getPresetArg(key)?.iniKey ?? key.replace(/_/g, '-');
}
