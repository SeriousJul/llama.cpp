import type { PresetData, PresetFieldState } from '$lib/utils/preset-utils';

export type { PresetData, PresetCustomKey, PresetFieldState };

export interface SettingsPresetSectionProps {
	presetName: string;
}

export interface SettingsPresetFieldProps {
	fieldName: string;
	presetName: string;
}

export interface SettingsPresetsCustomKeysProps {
	presetName: string;
}
