import { type ServerPresetOption, setPresetArgsCatalog } from '$lib/constants/preset-args-catalog';
import { parsePresetIni, serializePresetIni } from '$lib/utils/preset-utils';
import { beforeEach, describe, expect, it } from 'vitest';

const options: ServerPresetOption[] = [
	{
		args: ['--context-shift'],
		description: 'Context shift strategy.',
		key: 'context-shift',
		sampling: false,
		speculative: false,
		type: 'string',
		value_hint: 'STRATEGY'
	},
	{
		args: ['--prio'],
		description: 'Process priority.',
		key: 'prio',
		sampling: false,
		speculative: false,
		type: 'number',
		value_hint: 'N'
	}
];

describe('preset INI parsing', () => {
	beforeEach(() => setPresetArgsCatalog(options));

	it('loads the global section and recognizes canonical dashed keys', () => {
		const preset = parsePresetIni('[*]\ncontext-shift = false\nprio = 2\n');

		expect(preset.name).toBe('*');
		expect(preset.fields).toEqual({
			context_shift: { original: 'false', value: 'false' },
			prio: { original: '2', value: '2' }
		});
		expect(preset.customKeys).toEqual([]);
	});

	it('serializes normalized field names with their canonical INI keys', () => {
		const preset = parsePresetIni('[*]\ncontext-shift = false\nprio = 2\n');

		expect(serializePresetIni(preset)).toBe('[*]\ncontext-shift = false\nprio = 2\n\n');
	});
});
