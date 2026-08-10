<script lang="ts">
	import { ChevronDown, ChevronRight, Plus, Trash2 } from '@lucide/svelte';

	import {
		PRESET_ARGS_CATALOG,
		PRESET_CATEGORIES,
		PRESET_CATEGORY_ORDER,
		type PresetArgCatalogEntry,
		type PresetCategory
	} from '$lib/constants/preset-args-catalog';
	import { presetsStore } from '$lib/stores/presets.svelte';
	import { Input } from '$lib/components/ui/input';
	import SettingsPresetsCustomKeys from './SettingsPresetsCustomKeys.svelte';
	import SettingsPresetsField from './SettingsPresetsField.svelte';

	interface Props {
		presetName: string;
	}

	interface FieldGroup {
		category: PresetCategory;
		entries: PresetArgCatalogEntry[];
	}

	let { presetName }: Props = $props();

	const preset = $derived(presetsStore.getPreset(presetName));
	const isGlobal = $derived(preset?.source === 'global');
	const isNew = $derived(preset?.isNew === true);
	const dirtyCount = $derived.by(() => {
		if (!preset) return 0;
		let count = 0;
		if (preset.originalFieldCount !== undefined && Object.keys(preset.fields).length !== preset.originalFieldCount) {
			count += 1;
		}
		for (const field of Object.values(preset.fields)) {
			if (field.value !== field.original) count++;
		}
		return count;
	});

let collapsed = $state(true);
	let editingName = $state(false);
	let nameInput = $state('');

	function groupFields(entries: PresetArgCatalogEntry[]): FieldGroup[] {
		return PRESET_CATEGORY_ORDER.map((category) => ({
			category,
			entries: entries.filter((entry) => entry.category === category)
		})).filter((group) => group.entries.length > 0);
	}

	const visibleGroups = $derived.by(() => {
		if (!preset) return [];
		return groupFields(
			PRESET_ARGS_CATALOG.filter((entry) => !entry.hidden && preset.fields[entry.key])
		);
	});
	const emptyGroups = $derived.by(() => {
		if (!preset) return [];
		return groupFields(
			PRESET_ARGS_CATALOG.filter((entry) => !entry.hidden && !preset.fields[entry.key])
		);
	});

	function startEditName() {
		nameInput = preset?.name ?? presetName;
		editingName = true;
	}

	function finishEditName() {
		const nextName = nameInput.trim();
		if (nextName && nextName !== presetName) presetsStore.updatePresetName(presetName, nextName);
		editingName = false;
	}

	function cancelEditName() {
		nameInput = preset?.name ?? presetName;
		editingName = false;
	}

	function handleDelete() {
		if (confirm(`Delete preset "${presetName}"?`)) presetsStore.deletePreset(presetName);
	}

	function addField(entry: PresetArgCatalogEntry) {
		const target = presetsStore.getPreset(presetName);
		if (!target) return;
		target.fields[entry.key] = { value: entry.defaultValue ?? '', original: '' };
	}
</script>

<div class="rounded-lg border border-border/30" data-preset-card={presetName}>
	<div class="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-border/30 bg-background/95 backdrop-blur p-4 hover:bg-muted/50">
		<div class="flex min-w-0 items-center gap-2">
			<button
				type="button"
				aria-label={collapsed ? `Expand ${presetName}` : `Collapse ${presetName}`}
				class="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded hover:bg-muted"
				onclick={() => (collapsed = !collapsed)}
			>
				{#if collapsed}
					<ChevronRight class="h-4 w-4 text-muted-foreground" />
				{:else}
					<ChevronDown class="h-4 w-4 text-muted-foreground" />
				{/if}
			</button>

			{#if editingName}
				<Input
					bind:value={nameInput}
					onkeydown={(event) => {
						if (event.key === 'Enter') finishEditName();
						if (event.key === 'Escape') cancelEditName();
					}}
					onblur={finishEditName}
					class="font-semibold"
				/>
			{:else if isGlobal}
				<span class="truncate text-base font-semibold">{presetName}</span>
			{:else}
				<button type="button" class="truncate text-base font-semibold" onclick={startEditName}
					>{presetName}</button
				>
			{/if}

			<span class="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground"
				>{isGlobal ? 'global' : isNew ? 'new' : 'preset'}</span
			>
			{#if dirtyCount > 0}
				<span class="rounded bg-yellow-100 px-1.5 py-0.5 text-xs text-yellow-700"
					>{dirtyCount} modified</span
				>
			{/if}
		</div>

		{#if !isGlobal}
			<button
				type="button"
				aria-label={`Delete ${presetName}`}
				onclick={handleDelete}
				class="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded transition-colors hover:bg-muted"
			>
				<Trash2 class="h-4 w-4 text-muted-foreground" />
			</button>
		{/if}
	</div>

	{#if !collapsed}
		<div class="space-y-6 border-t border-border/30 p-4">
			{#each visibleGroups as group (group.category)}
				<section class="space-y-4 rounded-lg border border-border/40 bg-muted/10 p-4">
					<div>
						<h3 class="text-sm font-semibold">{PRESET_CATEGORIES[group.category].label}</h3>
						<p class="text-xs text-muted-foreground">
							{PRESET_CATEGORIES[group.category].description}
						</p>
					</div>
					<div class="space-y-5">
						{#each group.entries as entry (entry.key)}
							<SettingsPresetsField fieldName={entry.key} {presetName} />
						{/each}
					</div>
				</section>
			{/each}

			<SettingsPresetsCustomKeys {presetName} />

			{#if emptyGroups.length > 0}
				<details class="space-y-3 rounded-lg border border-dashed border-border/60 p-4">
					<summary class="cursor-pointer text-sm font-medium text-muted-foreground"
						>Add model parameters ({PRESET_ARGS_CATALOG.length -
							visibleGroups.reduce((count, group) => count + group.entries.length, 0)})</summary
					>
					<div class="space-y-3 pt-2">
						{#each emptyGroups as group (group.category)}
							<details class="rounded-md border border-border/40 bg-background p-3">
								<summary class="cursor-pointer text-sm font-medium"
									>{PRESET_CATEGORIES[group.category].label}
									<span class="text-xs font-normal text-muted-foreground"
										>({group.entries.length})</span
									></summary
								>
								<div class="grid gap-2 pt-3 md:grid-cols-2">
									{#each group.entries as entry (entry.key)}
										<button
											type="button"
											onclick={() => addField(entry)}
											class="flex h-auto items-start gap-2 rounded-md border border-input bg-background p-3 text-left text-xs transition-colors hover:bg-accent"
										>
											<Plus class="mt-0.5 h-3.5 w-3.5 shrink-0" />
											<span class="min-w-0 space-y-1">
												<span class="flex flex-wrap items-center gap-1.5 font-medium"
													><span>{entry.label}</span><code
														class="rounded bg-muted px-1 py-0.5 font-mono text-[10px]"
														>{entry.iniKey}</code
													></span
												>
												<span class="block text-muted-foreground">{entry.description}</span>
											</span>
										</button>
									{/each}
								</div>
							</details>
						{/each}
					</div>
				</details>
			{/if}
		</div>
	{/if}
</div>
