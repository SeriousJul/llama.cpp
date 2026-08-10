import { ModelsService } from '$lib/services/models.service';
import { presetsStore } from '$lib/stores/presets.svelte';
import type { ApiRouterModelsListResponse } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const presetOptions = [
	{
		args: ['--context-shift'],
		description: 'Whether to use context shift.',
		key: 'context-shift',
		sampling: false,
		speculative: false,
		type: 'boolean' as const,
		value_hint: ''
	},
	{
		args: ['--parallel'],
		description: 'Number of server slots.',
		key: 'parallel',
		sampling: false,
		speculative: false,
		type: 'number' as const,
		value_hint: 'N'
	}
];

describe('presetsStore', () => {
	beforeEach(() => vi.restoreAllMocks());

	it('loads the raw global section first and only configured model sections', async () => {
		const listRouter = vi.spyOn(ModelsService, 'listRouter').mockResolvedValue({
			data: [],
			object: 'list',
			preset_options: presetOptions,
			preset_sections: ['[*]\ncontext-shift = false\n\n', '[configured]\nparallel = 2\n\n']
		} as ApiRouterModelsListResponse);

		await presetsStore.initialize();

		expect(listRouter).toHaveBeenCalledWith(true);
		expect(presetsStore.getGlobalPreset()?.fields.context_shift.value).toBe('false');
		expect(presetsStore.getPreset('*')).toBe(presetsStore.getGlobalPreset());
		expect(presetsStore.getPresets().map((preset) => preset.name)).toEqual(['configured']);
		expect(presetsStore.getPreset('configured')?.fields.parallel.value).toBe('2');
		expect(presetsStore.hasLocalChanges()).toBe(false);

		presetsStore.updateFieldName('*', 'context_shift', 'true');
		expect(presetsStore.hasLocalChanges()).toBe(true);
	});
});
