/** Parse and serialize preset INI data to/from structured objects. */

import {
	getIniKey,
	getPresetArg,
	isKnownPresetArg,
	normalizeKey
} from '$lib/constants/preset-args-catalog';

export interface PresetFieldState {
	value: string;
	original: string;
}

export interface PresetCustomKey {
	key: string;
	value: string;
}

export interface PresetData {
	name: string;
	source?: PresetSource;
	fields: Record<string, PresetFieldState>;
	customKeys: PresetCustomKey[];
	isNew?: boolean;
	originalFieldCount?: number;
}

export type PresetSource = 'global' | 'custom-preset' | 'cached-model' | 'local-model';

export function parsePresetIni(ini: string, source?: PresetSource): PresetData {
	const lines = ini.split('\n');
	const nameMatch = lines[0]?.trim().match(/^\[(.+)\]$/);
	const name = nameMatch?.[1] ?? 'unnamed';
	const fields: Record<string, PresetFieldState> = {};
	const customKeys: PresetCustomKey[] = [];

	for (let i = 1; i < lines.length; i++) {
		const line = lines[i].trim();
		if (!line || line.startsWith('[')) continue;

		const eqIdx = line.indexOf('=');
		if (eqIdx === -1) continue;

		const rawKey = line.slice(0, eqIdx).trim();
		const value = line.slice(eqIdx + 1).trim();
		if (!rawKey) continue;

		const key = normalizeKey(rawKey);
		if (isKnownPresetArg(key)) {
			fields[key] = { value, original: value };
		} else {
			customKeys.push({ key: rawKey, value });
		}
	}

	return { name, source, fields, customKeys };
}

export function serializePresetIni(preset: PresetData): string {
	let result = `[${preset.name}]\n`;

	for (const [key, field] of Object.entries(preset.fields)) {
		const catalogEntry = getPresetArg(key);
		const iniKey = getIniKey(key);

		if (catalogEntry?.isList) {
			const items = field.value
				.split(',')
				.map((item) => item.trim())
				.filter(Boolean);
			result += `${iniKey} = ${items.join(',')}\n`;
		} else if (catalogEntry?.type === 'boolean') {
			const normalized = field.value.toLowerCase();
			if (['true', 'yes', '1', 'on'].includes(normalized)) {
				result += `${iniKey} = true\n`;
			} else if (['false', 'no', '0', 'off'].includes(normalized)) {
				result += `${iniKey} = false\n`;
			} else {
				result += `${iniKey} = ${field.value}\n`;
			}
		} else {
			result += `${iniKey} = ${field.value}\n`;
		}
	}

	for (const customKey of preset.customKeys) {
		result += `${customKey.key} = ${customKey.value}\n`;
	}

	return result + '\n';
}

export function isFieldDirty(field: PresetFieldState): boolean {
	return field.value !== field.original;
}

export function generateFullIni(globalPreset: PresetData | null, presets: PresetData[]): string {
	let result = '';

	if (globalPreset) result += serializePresetIni(globalPreset);
	for (const preset of presets) result += serializePresetIni(preset);

	return result;
}
