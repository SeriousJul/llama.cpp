import { setPresetArgsCatalog, type ServerPresetOption } from '$lib/constants/preset-args-catalog';
import { parsePresetIni, serializePresetIni } from '$lib/utils/preset-utils';
import { beforeEach, describe, expect, it } from 'vitest';

const options: ServerPresetOption[] = [
	{
		key: 'context-shift',
		args: ['--context-shift'],
		value_hint: 'STRATEGY',
		description: 'Context shift strategy.',
		type: 'string',
		sampling: false,
		speculative: false
	},
	{
		key: 'prio',
		args: ['--prio'],
		value_hint: 'N',
		description: 'Process priority.',
		type: 'number',
		sampling: false,
		speculative: false
	}
];

describe('preset INI parsing', () => {
	beforeEach(() => setPresetArgsCatalog(options));

	it('loads the global section and recognizes canonical dashed keys', () => {
		const preset = parsePresetIni('[*]\ncontext-shift = false\nprio = 2\n');

		expect(preset.name).toBe('*');
		expect(preset.fields).toEqual({
			context_shift: { value: 'false', original: 'false' },
			prio: { value: '2', original: '2' }
		});
		expect(preset.customKeys).toEqual([]);
	});

	it('serializes normalized field names with their canonical INI keys', () => {
		const preset = parsePresetIni('[*]\ncontext-shift = false\nprio = 2\n');
		expect(serializePresetIni(preset)).toBe('[*]\ncontext-shift = false\nprio = 2\n\n');
	});
});
