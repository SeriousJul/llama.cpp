<script lang="ts">
	import { Download, List, Plus, RefreshCw, Settings2 } from '@lucide/svelte';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { Button } from '$lib/components/ui/button';
	import SettingsPresetsSection from './SettingsPresetsSection.svelte';
	import { ModelsService } from '$lib/services/models.service';
	import { presetsStore } from '$lib/stores/presets.svelte';
	import type { PresetData } from '$lib/types/presets';

	onMount(() => {
		presetsStore.initialize();
	});

	const presets = $derived(presetsStore.getPresets());
	const globalPreset = $derived(presetsStore.getGlobalPreset());
	const dirtyCount = $derived(presetsStore.getDirtyCount());

	let showOutline = $state(false);
	let isReloading = $state(false);

	function createPreset() {
		const name = `preset-${presets.length + 1}`;

		presetsStore.addPreset({
			name,
			source: 'custom-preset',
			fields: {},
			customKeys: [],
			isNew: true
		});
	}

	function handleDownload() {
		presetsStore.downloadIni();
	}

	async function handleReload() {
		if (
			presetsStore.hasLocalChanges() &&
			!confirm('Reload presets from the server and discard local changes?')
		)
			return;

		isReloading = true;
		try {
			await ModelsService.reload();
			await presetsStore.initialize();
			toast.success('Presets reloaded from the INI file');
		} catch (error) {
			console.error('Failed to reload presets:', error);
			toast.error('Failed to reload presets');
		} finally {
			isReloading = false;
		}
	}

	function scrollToPreset(name: string) {
		const element = [...document.querySelectorAll<HTMLElement>('[data-preset-card]')].find(
			(candidate) => candidate.dataset.presetCard === name
		);
		element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}

	const allPresets = $derived([globalPreset, ...presets].filter(Boolean) as PresetData[]);
</script>

<div class="space-y-6">
	<div class="flex flex-wrap items-center justify-between gap-3">
		<div class="flex items-center gap-2">
			<Settings2 class="h-5 w-5" />

			<h2 class="text-lg font-semibold">Presets</h2>

			{#if dirtyCount > 0}
				<span class="text-xs px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700"
					>{dirtyCount} modified</span
				>
			{/if}
		</div>

		<div class="flex flex-wrap items-center justify-end gap-2">
			<Button size="sm" variant="outline" onclick={() => (showOutline = !showOutline)}>
				<List class="h-4 w-4" />

				{showOutline ? 'Hide' : 'Show'} outline
			</Button>

			<Button size="sm" variant="outline" onclick={handleReload} disabled={isReloading}>
				<RefreshCw class={isReloading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />

				{isReloading ? 'Reloading...' : 'Reload INI'}
			</Button>

			<Button size="sm" variant="outline" onclick={handleDownload}>
				<Download class="h-4 w-4" />

				Download INI
			</Button>

			<Button size="sm" onclick={createPreset}>
				<Plus class="h-4 w-4" />

				New Preset
			</Button>
		</div>
	</div>

	{#if showOutline && allPresets.length > 0}
		<div class="rounded-lg border border-border/30 bg-muted/30 p-4">
			<p class="mb-2 text-xs font-medium uppercase text-muted-foreground">Navigate presets</p>

			<div class="flex flex-wrap gap-2">
				{#each allPresets as preset (preset.name)}
					<button
						type="button"
						onclick={() => scrollToPreset(preset.name)}
						class="inline-flex h-7 items-center rounded-md border border-input bg-background px-3 text-xs transition-colors hover:bg-accent"
						>{preset.name}</button
					>
				{/each}
			</div>
		</div>
	{/if}

	{#if allPresets.length === 0}
		<div class="rounded-lg border border-border/30 p-8 text-center text-muted-foreground">
			<p class="text-sm">No presets found. Create one to get started.</p>

			<p class="text-xs mt-1">Presets are loaded from the server's presets.ini file.</p>
		</div>
	{:else}
		<div class="space-y-4">
			{#each allPresets as preset (preset.name)}
				<SettingsPresetsSection presetName={preset.name} />
			{/each}
		</div>
	{/if}

	<p class="text-xs text-muted-foreground">
		Changes are local until you download and apply the INI file to the server.
	</p>
</div>
