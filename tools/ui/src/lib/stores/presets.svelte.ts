import { setPresetArgsCatalog } from '$lib/constants/preset-args-catalog';
import { ModelsService } from '$lib/services/models.service';
import type { PresetData } from '$lib/types/presets';
import { generateFullIni, isFieldDirty, parsePresetIni } from '$lib/utils/preset-utils';

let presets: PresetData[] = $state([]);
let globalPreset: PresetData | null = $state(null);
let loadedIni = '';

function emptyGlobalPreset(): PresetData {
	return { name: '*', source: 'global', fields: {}, customKeys: [] };
}

export const presetsStore = {
	async initialize() {
		try {
			const response = await ModelsService.listRouter(true);
			setPresetArgsCatalog(response.preset_options ?? []);

			let nextGlobalPreset: PresetData | null = null;
			const nextPresets: PresetData[] = [];

		for (const ini of response.preset_sections ?? []) {
			const parsed = parsePresetIni(ini);
			const fieldCount = Object.keys(parsed.fields).length;
			parsed.originalFieldCount = fieldCount;
			if (parsed.name === '*' || parsed.name === 'global') {
				nextGlobalPreset = { ...parsed, name: '*', source: 'global' };
			} else {
				nextPresets.push({ ...parsed, source: 'custom-preset' });
			}
		}

			globalPreset = nextGlobalPreset ?? emptyGlobalPreset();
			presets = nextPresets;
			loadedIni = generateFullIni(globalPreset, presets);
		} catch (error) {
			globalPreset ??= emptyGlobalPreset();
			console.error('Failed to load presets:', error);
		}
	},

	getPresets(): PresetData[] {
		return presets;
	},

	getGlobalPreset(): PresetData | null {
		return globalPreset;
	},

	removeField(presetName: string, fieldName: string) {
		const preset = this.getPreset(presetName);
		if (preset?.fields[fieldName]) {
			const { [fieldName]: _, ...rest } = preset.fields;
			preset.fields = rest;
		}
	},

	updateFieldName(presetName: string, fieldName: string, value: string) {
		const preset = this.getPreset(presetName);
		if (preset?.fields[fieldName]) preset.fields[fieldName].value = value;
	},

	updatePresetName(oldName: string, newName: string) {
		const preset = presets.find((item) => item.name === oldName);
		if (preset) preset.name = newName;
	},

	addPreset(preset: PresetData) {
		presets.push(preset);
	},

	deletePreset(name: string) {
		presets = presets.filter((preset) => preset.name !== name);
	},

	getPreset(name: string): PresetData | undefined {
		if (globalPreset?.name === name) return globalPreset;
		return presets.find((preset) => preset.name === name);
	},

	hasLocalChanges(): boolean {
		return generateFullIni(globalPreset, presets) !== loadedIni;
	},

	getDirtyCount(): number {
		let count = 0;
		for (const preset of [...presets, globalPreset].filter(Boolean) as PresetData[]) {
			if (preset.originalFieldCount !== undefined && Object.keys(preset.fields).length !== preset.originalFieldCount) {
				count += 1;
			}
			for (const field of Object.values(preset.fields)) {
				if (isFieldDirty(field)) count++;
			}
		}
		return count;
	},

	downloadIni() {
		const ini = generateFullIni(globalPreset, presets);
		const blob = new Blob([ini], { type: 'text/plain' });
		const url = URL.createObjectURL(blob);
		const anchor = document.createElement('a');

		anchor.href = url;
		anchor.download = 'presets.ini';
		anchor.click();
		URL.revokeObjectURL(url);
	}
};
