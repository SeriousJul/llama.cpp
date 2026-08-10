<script lang="ts">
	import { Info, RotateCcw, Trash2 } from '@lucide/svelte';

	import * as Select from '$lib/components/ui/select';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { getPresetArg } from '$lib/constants/preset-args-catalog';
	import { presetsStore } from '$lib/stores/presets.svelte';

	interface Props {
		fieldName: string;
		presetName: string;
	}

	let { fieldName, presetName }: Props = $props();

	const catalogEntry = $derived(getPresetArg(fieldName));
	const preset = $derived(presetsStore.getPreset(presetName));
	const fieldState = $derived(preset?.fields[fieldName]);
	const isDirty = $derived(fieldState ? fieldState.value !== fieldState.original : false);

	function handleInput(event: Event) {
		presetsStore.updateFieldName(presetName, fieldName, (event.target as HTMLInputElement).value);
	}

	function handleReset() {
		if (fieldState) {
			presetsStore.updateFieldName(presetName, fieldName, fieldState.original);
		}
	}

	function handleRemove() {
		presetsStore.removeField(presetName, fieldName);
	}

	function parseListItems(value: string): string[] {
		return value
			.split(',')
			.map((s) => s.trim())
			.filter(Boolean);
	}

	function removeListItem(index: number) {
		const items = parseListItems(fieldState?.value ?? '');

		items.splice(index, 1);

		presetsStore.updateFieldName(presetName, fieldName, items.join(','));
	}

	function addListItem(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			event.preventDefault();
			const value = (event.currentTarget as HTMLInputElement).value.trim();

			if (value) {
				const current = fieldState?.value ?? '';
				presetsStore.updateFieldName(
					presetName,
					fieldName,
					`${current},${value}`.replace(/^,/, '')
				);

				(event.currentTarget as HTMLInputElement).value = '';
			}
		}
	}

	const fieldValue = $derived(fieldState?.value ?? '');

	const isBoolean = $derived(catalogEntry?.type === 'boolean');
	const isNumber = $derived(catalogEntry?.type === 'number');
	const isListString = $derived(catalogEntry?.type === 'list-string');

	function getCheckboxChecked(): boolean {
		if (!fieldState) return false;

		return ['true', 'yes', '1', 'on'].includes(fieldState.value.toLowerCase());
	}

	function toggleCheckbox(checked: boolean) {
		presetsStore.updateFieldName(presetName, fieldName, checked ? 'true' : 'false');
	}

	const selectOptions = $derived(() => {
		if (catalogEntry?.help) {
			const match = catalogEntry.help.match(/"([^"]+)"/g);

			if (match) {
				return match.map((m) => ({ value: m.replace(/"/g, ''), label: m.replace(/"/g, '') }));
			}
		}

		return [];
	});

	const hasSelectOptions = $derived(selectOptions().length > 0);

	function handleSelectChange(value: string) {
		presetsStore.updateFieldName(presetName, fieldName, value);
	}

	const listItems = $derived(parseListItems(fieldValue));
</script>

<div class="space-y-2">
	<div class="flex flex-wrap items-center gap-2">
		<Label for={`preset-field-${presetName}-${fieldName}`} class="text-sm font-medium">
			{catalogEntry?.label ?? fieldName}

			{#if isDirty}
				<span class="ml-1 text-xs text-yellow-600" title="Modified from original">*</span>
			{/if}
		</Label>

		<code class="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
			>{catalogEntry?.iniKey ?? fieldName} = {fieldValue || '""'}</code
		>

		{#if isDirty}
			<button
				type="button"
				onclick={handleReset}
				class="inline-flex h-5 w-5 items-center justify-center rounded transition-colors hover:bg-muted"
				title="Reset to original value"
			>
				<RotateCcw class="h-3 w-3 text-muted-foreground" />
			</button>
		{/if}

		<button
			type="button"
			onclick={handleRemove}
			class="inline-flex h-5 w-5 items-center justify-center rounded transition-colors hover:bg-muted"
			title="Remove parameter"
		>
			<Trash2 class="h-3 w-3 text-muted-foreground" />
		</button>

		{#if catalogEntry?.help}
			<div class="group relative">
				<Info class="h-3.5 w-3.5 cursor-help text-muted-foreground" />

				<div
					class="absolute z-10 hidden w-64 rounded-md border bg-popover p-3 text-xs shadow-md group-hover:block"
				>
					{catalogEntry.help}
				</div>
			</div>
		{/if}
	</div>

	{#if catalogEntry?.description}
		<p class="text-xs text-muted-foreground">{catalogEntry.description}</p>
	{/if}

	{#if isBoolean}
		<Checkbox
			id={`preset-field-${presetName}-${fieldName}`}
			class="mt-1"
			checked={getCheckboxChecked()}
			onCheckedChange={toggleCheckbox}
		/>
	{:else if isNumber || catalogEntry?.type === 'string'}
		<Input
			id={`preset-field-${presetName}-${fieldName}`}
			type={isNumber ? 'number' : 'text'}
			{...isNumber ? { step: '1' } : {}}
			value={fieldValue}
			oninput={handleInput}
		/>
	{:else if isListString}
		<div class="flex flex-wrap gap-2">
			{#each listItems as item, index (item)}
				<span class="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs">
					{item}

					<button type="button" onclick={() => removeListItem(index)}>x</button>
				</span>
			{/each}

			<Input
				id={`preset-field-${presetName}-${fieldName}`}
				placeholder="Add item"
				onkeydown={addListItem}
				class="w-32"
			/>
		</div>
	{:else if hasSelectOptions}
		<Select.Root type="single" value={fieldValue} onValueChange={handleSelectChange}>
			<Select.Trigger class="w-full">
				{selectOptions().find((opt) => opt.value === fieldValue)?.label ?? 'Select...'}
			</Select.Trigger>

			<Select.Content>
				{#each selectOptions() as option (option.value)}
					<Select.Item value={option.value} label={option.label}>{option.label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	{:else}
		<Input
			id={`preset-field-${presetName}-${fieldName}`}
			value={fieldValue}
			oninput={handleInput}
		/>
	{/if}
</div>
