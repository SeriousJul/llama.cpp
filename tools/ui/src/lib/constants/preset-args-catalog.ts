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
	model: {
		label: 'Model source',
		description: 'Model files, repositories, aliases, and metadata.'
	},
	compute: { label: 'CPU and GPU', description: 'Thread scheduling, devices, and tensor offload.' },
	context: { label: 'Context', description: 'Context size, shifting, and long-context behavior.' },
	batching: {
		label: 'Batching and parallelism',
		description: 'Batch sizes, slots, and parallel decoding.'
	},
	memory: {
		label: 'Loading and memory',
		description: 'Model loading, mapping, locking, and placement.'
	},
	'kv-cache': {
		label: 'KV cache',
		description: 'KV cache formats, offload, reuse, and checkpoints.'
	},
	sampling: {
		label: 'Sampling',
		description: 'Token selection, penalties, grammars, and generation limits.'
	},
	rope: { label: 'RoPE and YaRN', description: 'Position scaling and context extension.' },
	multimodal: { label: 'Multimodal', description: 'Projectors and image or media processing.' },
	embedding: {
		label: 'Embedding and reranking',
		description: 'Pooling, embedding, and reranking behavior.'
	},
	adapters: {
		label: 'Adapters',
		description: 'LoRA adapters, control vectors, and tensor overrides.'
	},
	chat: { label: 'Chat and tools', description: 'Templates, reasoning, tools, MCP, and agents.' },
	server: {
		label: 'Server runtime',
		description: 'HTTP behavior, endpoints, timeouts, and Web UI options.'
	},
	speculative: {
		label: 'Speculative decoding',
		description: 'Draft models and speculative decoding algorithms.'
	},
	router: { label: 'Router lifecycle', description: 'Per-model loading and shutdown behavior.' },
	logging: { label: 'Logging', description: 'Logging output and diagnostics.' },
	other: { label: 'Other', description: 'Additional server-supported model options.' }
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
	context: new Set([
		'ctx-size',
		'n-predict',
		'keep',
		'context-shift',
		'swa-full',
		'swa-checkpoints',
		'checkpoint-min-step'
	]),
	batching: new Set(['batch-size', 'ubatch-size', 'parallel', 'cont-batching']),
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
	embedding: new Set(['pooling', 'embeddings', 'reranking', 'embd-normalize']),
	adapters: new Set([
		'lora',
		'lora-scaled',
		'lora-init-without-apply',
		'control-vector',
		'control-vector-scaled',
		'override-kv',
		'override-tensor'
	]),
	server: new Set(['threads-http']),
	router: new Set(['load-on-startup', 'stop-timeout'])
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
			key: normalizeKey(option.key),
			iniKey: option.key,
			label: labelFor(option.key),
			description: option.description.trim(),
			help: option.description.trim(),
			type,
			category: categoryFor(option),
			args: option.args,
			valueHint: option.value_hint || undefined,
			isList,
			defaultValue: type === 'boolean' ? 'false' : ''
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
